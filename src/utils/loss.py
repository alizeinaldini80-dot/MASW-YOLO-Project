"""
src/utils/loss.py

پیاده‌سازی Wise-IoU نسخه ۳ (WIoU-v3، Tong et al. 2023) به‌عنوان جایگزین
CIoU در محاسبهٔ Bounding-Box Regression Loss، طبق بخش مربوطه در مقاله
MASW-YOLO.

چرا WIoU؟
    CIoU/GIoU/DIoU با فرض این‌که همهٔ باکس‌های anchor باکیفیت‌اند، جریمهٔ
    ثابتی برای فاصله/نسبت‌ابعاد اعمال می‌کنند. این باعث می‌شود روی
    داده‌های noisy یا دارای باکس‌های کم‌کیفیت (مثل VisDrone که خیلی از
    جعبه‌ها ریز و مبهم‌اند)، مدل به‌اندازهٔ کافی روی نمونه‌های "متوسط"
    تمرکز نکند. WIoU-v3 با یک مکانیزم تمرکز غیریکنواخت (non-monotonic
    focusing) وزن هر نمونه را بر اساس "درجهٔ پرت‌بودنش" (outlier degree)
    نسبت به میانگین متحرک IoU کل batch ها تنظیم می‌کند: نه به نمونه‌های
    خیلی آسان بیش‌ازحد وزن می‌دهد، نه به نمونه‌های خیلی بد (که احتمالاً
    برچسب نویزی دارند) بیش‌ازحد اهمیت می‌دهد.

این ماژول یک monkey-patch سراسری روی ultralytics.utils.loss.BboxLoss
انجام می‌دهد (دقیقاً مثل الگوی src/models/parsing.py و src/utils/nms.py):
    from src.utils.loss import enable_wiou
    enable_wiou()
    # از این‌جا به بعد، v8DetectionLoss (که توسط model.train() و
    # model.val() به‌طور خودکار ساخته می‌شود) از WIoU-v3 به‌جای CIoU
    # استفاده می‌کند.

⚠️ نکتهٔ نسخه: ساختار BboxLoss.forward در این فایل عیناً از سورس نصب‌شدهٔ
ultralytics.utils.loss.BboxLoss کپی شده (فقط بخش محاسبهٔ iou/loss_iou
عوض شده). اگر بعد از آپدیت نسخه در کولب خطای غیرمنتظره گرفتید:
    import inspect, ultralytics.utils.loss as l
    print(inspect.getsource(l.BboxLoss))
و با این فایل مقایسه کنید.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import bbox2dist


def _box_xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def _iou(boxes1, boxes2):
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-7)


def _center_distance(boxes1, boxes2):
    c1 = _box_xyxy_to_cxcywh(boxes1)
    c2 = _box_xyxy_to_cxcywh(boxes2)
    return (c1[..., :2] - c2[..., :2]).pow(2).sum(dim=-1).sqrt()


def wise_iou(boxes1, boxes2, version="v3", beta=1.0, delta=0.5):
    iou = _iou(boxes1, boxes2)
    loss_iou = 1.0 - iou
    if version == "v1":
        return loss_iou.unsqueeze(-1)
    c_dist = _center_distance(boxes1, boxes2)
    enclose_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    enclose_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    enclose_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    enclose_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    enclose_diag = torch.sqrt((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).clamp(min=1e-7)
    r_wiou = torch.exp((c_dist ** 2) / (enclose_diag ** 2))
    if version == "v3":
        with torch.no_grad():
            loss_mean = loss_iou.mean().detach()
            loss_std = loss_iou.std().detach() + 1e-7
            alpha = (loss_iou - loss_mean) / loss_std
            r = beta / (delta * alpha.clamp(min=1e-7) ** (delta - beta) + 1e-7)
            r = r.clamp(max=10.0)
        loss = r * r_wiou * loss_iou
        return loss.unsqueeze(-1)
    raise ValueError(f"Unknown WIoU version: {version}")


class WiseIoULoss(BboxLoss):
    def __init__(self, reg_max=16, version="v3", beta=1.0, delta=0.5):
        super().__init__(reg_max)
        self.version = version
        self.beta = beta
        self.delta = delta

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = wise_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], version=self.version, beta=self.beta, delta=self.delta)
        loss_iou = (iou * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight)
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl

# ---------------------------------------------------------------------------
# فعال‌سازی سراسری (Monkey-patch) — برای استفاده در ablation از طریق کانفیگ
# ---------------------------------------------------------------------------

import ultralytics.utils.loss as _ultra_loss
from ultralytics.utils import LOGGER


def enable_wiou(version: str = "v3", beta: float = 1.0, delta: float = 0.5):
    """
    فعال‌سازی سراسری WiseIoULoss: کلاس ultralytics.utils.loss.BboxLoss را با
    WiseIoULoss جایگزین می‌کند. چون v8DetectionLoss.__init__ نام `BboxLoss`
    را در زمان فراخوانی (نه زمان import) از namespace همین ماژول resolve
    می‌کند، این patch باید فقط قبل از اولین فراخوانی model.train()/model.val()
    (یعنی قبل از ساخته‌شدن criterion) اجرا شود.

    نحوه استفاده (در configs/exp*.yaml):
        wiou: true
        wiou_version: v3
        wiou_beta: 1.0
        wiou_delta: 0.5
    """
    def _factory(reg_max: int = 16, *args, **kwargs):
        return WiseIoULoss(reg_max, version=version, beta=beta, delta=delta)

    _ultra_loss.BboxLoss = _factory
    LOGGER.info(f"✅ WiseIoU ({version}) فعال شد (beta={beta}, delta={delta})")


def disable_wiou():
    """بازگرداندن CIoU/BboxLoss استاندارد Ultralytics (در صورت نیاز به مقایسه در همان اجرا)."""
    import importlib
    importlib.reload(_ultra_loss)
    LOGGER.info("↩️ BboxLoss استاندارد Ultralytics بازگردانده شد.")
