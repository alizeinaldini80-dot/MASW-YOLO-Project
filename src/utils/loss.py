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
import ultralytics.utils.loss as _ultra_loss
from ultralytics.utils import LOGGER


def wiou_v3(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    iou_mean: torch.Tensor,
    momentum: float = 0.9999,
    delta: float = 3.0,
    alpha: float = 1.9,
    eps: float = 1e-7,
):
    """
    محاسبهٔ IoU معمولی و Loss نسخهٔ WIoU-v3، طبق فرمول‌های (اصلی) مقالهٔ
    Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism.

    Args:
        pred_bboxes, target_bboxes (Tensor[N,4]): مختصات xyxy
        iou_mean (Tensor scalar, buffer): میانگین متحرک L_IoU کل آموزش
            (برای محاسبهٔ β و r؛ در جا [in-place] به‌روزرسانی می‌شود)
        momentum: ضریب میانگین متحرک (نزدیک به ۱ یعنی به‌کندی به‌روز می‌شود)
        delta, alpha: هایپرپارامترهای مکانیزم تمرکز غیریکنواخت WIoU-v3
        eps: برای پایداری عددی

    Returns:
        iou (Tensor[N]): IoU خام (فقط برای لاگ/دیباگ، بدون گرادیان)
        loss (Tensor[N]): مقدار Loss نهایی WIoU-v3 برای هر باکس (با گرادیان)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.unbind(-1)

    # اشتراک و اجتماع
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)
    w1, h1 = (b1_x2 - b1_x1), (b1_y2 - b1_y1)
    w2, h2 = (b2_x2 - b2_x1), (b2_y2 - b2_y1)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # قطر باکس محیطی (enclosing box) - طبق مقاله WIoU از گرادیان جدا می‌شود
    # تا از کند شدن همگرایی توسط این جمله جلوگیری شود
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = (cw.pow(2) + ch.pow(2)).detach() + eps

    # فاصلهٔ مرکز دو باکس
    center_dist2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2).pow(2) + (b1_y1 + b1_y2 - b2_y1 - b2_y2).pow(2)) / 4

    r_wiou = torch.exp(center_dist2 / c2)  # جریمهٔ فاصله (WIoU-v1)
    l_iou = 1.0 - iou
    l_wiou_v1 = r_wiou * l_iou

    # به‌روزرسانی میانگین متحرک L_IoU (بدون گرادیان) برای محاسبهٔ β
    with torch.no_grad():
        if l_iou.numel() > 0:
            batch_mean = l_iou.mean()
            iou_mean.mul_(momentum).add_(batch_mean * (1.0 - momentum))

    # درجهٔ پرت‌بودن (outlier degree) و ضریب تمرکز غیریکنواخت r (بدون گرادیان)
    beta = (l_iou.detach() / (iou_mean + eps)).clamp(min=0.0)
    r = beta / (delta * torch.pow(torch.as_tensor(alpha, device=beta.device, dtype=beta.dtype), beta - delta))

    loss = r.detach() * l_wiou_v1
    return iou.detach(), loss


class WIoUBboxLoss(nn.Module):
    """
    جایگزین BboxLoss اصلی Ultralytics؛ همان ورودی/خروجی را دارد (تا
    v8DetectionLoss بدون تغییر با آن کار کند)، فقط IoU loss را با
    WIoU-v3 به‌جای CIoU محاسبه می‌کند. DFL loss دقیقاً مثل نسخهٔ اصلی است.
    """

    def __init__(self, reg_max: int = 16, wiou_delta: float = 3.0, wiou_alpha: float = 1.9,
                 wiou_momentum: float = 0.9999):
        super().__init__()
        self.dfl_loss = _ultra_loss.DFLoss(reg_max) if reg_max > 1 else None
        self.wiou_delta = wiou_delta
        self.wiou_alpha = wiou_alpha
        self.wiou_momentum = wiou_momentum
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        """محاسبهٔ WIoU-v3 loss و DFL loss (ساختار عیناً مطابق BboxLoss اصلی)."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # ==== تنها تفاوت اصلی نسبت به BboxLoss اصلی Ultralytics ====
        _, wiou = wiou_v3(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], self.iou_mean,
            momentum=self.wiou_momentum, delta=self.wiou_delta, alpha=self.wiou_alpha,
        )
        loss_iou = (wiou.unsqueeze(-1) * weight).sum() / target_scores_sum
        # ==============================================================

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            import torch.nn.functional as F
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


def enable_wiou(delta: float = 3.0, alpha: float = 1.9, momentum: float = 0.9999):
    """
    فعال‌سازی سراسری WIoU-v3: کلاس ultralytics.utils.loss.BboxLoss را با
    WIoUBboxLoss جایگزین می‌کند. چون v8DetectionLoss.__init__ نام
    `BboxLoss` را در زمان فراخوانی (نه زمان import) از namespace همین
    ماژول resolve می‌کند، این patch باید فقط قبل از اولین فراخوانی
    model.train()/model.val() (یعنی قبل از ساخته‌شدن criterion) اجرا شود.
    """
    def _factory(reg_max=16, *args, **kwargs):
        return WIoUBboxLoss(reg_max, wiou_delta=delta, wiou_alpha=alpha, wiou_momentum=momentum)

    _ultra_loss.BboxLoss = _factory
    LOGGER.info(f"✅ WIoU-v3 فعال شد (delta={delta}, alpha={alpha}, momentum={momentum})")


def disable_wiou():
    """بازگرداندن CIoU/BboxLoss استاندارد Ultralytics (در صورت نیاز به مقایسه در همان اجرا)."""
    import importlib
    importlib.reload(_ultra_loss)
