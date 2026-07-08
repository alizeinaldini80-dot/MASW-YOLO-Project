"""
src/utils/nms.py

پیاده‌سازی Soft-NMS (Bodla et al., 2017) به‌عنوان جایگزین NMS سخت
(hard-NMS) استاندارد، طبق بخش مربوطه در مقاله MASW-YOLO.

نسخهٔ بهینه‌شده (v2):
    نسخهٔ اول یک حلقهٔ پایتونی بود که برای هر باکس، جداگانه IoU با
    باقیمانده‌ها را محاسبه می‌کرد. در validation با conf_thres پایین
    (۰.۰۰۱ طبق تنظیمات پیش‌فرض Ultralytics)، هزاران باکس کاندید قبل از
    NMS زنده می‌مانند و این حلقه به‌خاطر overhead هر فراخوانی جدا روی
    GPU به‌شدت کند می‌شود (چند ده ثانیه به‌ازای هر تصویر).

    راه‌حل: ماتریس IoU بین همهٔ جفت‌باکس‌ها فقط یک‌بار محاسبه می‌شود
    (N×N)، و حلقهٔ سرکوب فقط از این ماتریس ایندکس می‌خواند (بدون
    فراخوانی box_iou جدید در هر تکرار). علاوه‌بر این، برای این‌که ماتریس
    N×N از نظر حافظه منفجر نشود (۳۰٬۰۰۰ باکس یعنی ماتریسی با ۹۰۰ میلیون
    عضو!)، قبل از Soft-NMS فقط بالاترین soft_nms_max_boxes امتیاز
    نگه‌داشته می‌شود (پیش‌فرض ۱۰۰۰؛ باکس‌های خارج از این محدوده عملاً در
    NMS سخت هم شانسی برای زنده‌ماندن نداشتند).

استفاده:
    from src.utils.nms import enable_soft_nms
    enable_soft_nms(method="gaussian", sigma=0.5, score_thres=0.001)

⚠️ نکتهٔ نسخه: ساختار non_max_suppression_soft عیناً بر اساس
ultralytics.utils.nms.non_max_suppression (نسخهٔ نصب‌شده هنگام نوشتن این
کد) است. اگر بعد از آپدیت نسخه در کولب خطای غیرمنتظره گرفتید:
    import inspect, ultralytics.utils.nms as n
    print(inspect.getsource(n.non_max_suppression))
و با این فایل مقایسه کنید.
"""

import time

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

import ultralytics.utils.nms as _ultra_nms


def soft_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float = 0.5,
             sigma: float = 0.5, score_thres: float = 0.001, method: str = "gaussian",
             max_boxes: int = 1000):
    """
    هستهٔ الگوریتم Soft-NMS — نسخهٔ وکتوریزه (ماتریس IoU فقط یک‌بار محاسبه می‌شود).

    Args:
        boxes (Tensor[N,4]): مختصات xyxy
        scores (Tensor[N]): امتیاز اطمینان هر باکس
        iou_thres (float): آستانهٔ IoU (فقط برای method='linear' یا 'hard')
        sigma (float): پارامتر واریانس گاووسی (فقط برای method='gaussian')
        score_thres (float): آستانهٔ نهایی حذف باکس‌های امتیاز-کاهش‌یافته
        method (str): 'gaussian' | 'linear' | 'hard'
        max_boxes (int): حداکثر تعداد باکس ورودی به الگوریتم (برای کنترل
            حافظه/زمان ماتریس N×N؛ باکس‌های کم‌امتیازتر از این سقف از قبل
            حذف می‌شوند، دقیقاً مثل رفتار NMS سخت روی long-tail کم‌امتیاز)

    Returns:
        Tensor[K]: اندیس باکس‌های نگه‌داشته‌شده (نسبت به boxes/scores ورودی)
    """
    n = boxes.shape[0]
    if n == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    device = boxes.device

    # پیش‌غربالگری برای کنترل اندازهٔ ماتریس IoU
    if n > max_boxes:
        top = torch.argsort(scores, descending=True)[:max_boxes]
        work_boxes = boxes[top]
        work_scores = scores[top].clone()
    else:
        top = torch.arange(n, device=device)
        work_boxes = boxes
        work_scores = scores.clone()

    m = work_boxes.shape[0]
    iou_mat = box_iou(work_boxes, work_boxes)  # فقط یک‌بار محاسبه می‌شود (m×m)

    active = torch.ones(m, dtype=torch.bool, device=device)
    keep_local = []

    neg_inf = torch.finfo(work_scores.dtype).min

    for _ in range(m):
        masked = torch.where(active, work_scores, torch.full_like(work_scores, neg_inf))
        i = int(torch.argmax(masked))
        if work_scores[i] <= score_thres or not active[i]:
            break
        keep_local.append(i)
        active[i] = False
        if not active.any():
            break

        ious = iou_mat[i]
        if method == "linear":
            decay = torch.where(ious > iou_thres, 1.0 - ious, torch.ones_like(ious))
        elif method == "gaussian":
            decay = torch.exp(-(ious * ious) / sigma)
        elif method == "hard":
            decay = torch.where(ious > iou_thres, torch.zeros_like(ious), torch.ones_like(ious))
        else:
            raise ValueError(f"روش نامعتبر برای Soft-NMS: {method}")

        work_scores = torch.where(active, work_scores * decay, work_scores)
        active &= work_scores > score_thres

    if not keep_local:
        return torch.zeros((0,), dtype=torch.long, device=device)

    keep_local_t = torch.tensor(keep_local, dtype=torch.long, device=device)
    return top[keep_local_t]


def non_max_suppression_soft(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
    # --- پارامترهای مخصوص Soft-NMS ---
    soft_nms_method: str = "gaussian",
    soft_nms_sigma: float = 0.5,
    soft_nms_score_thres: float = 0.001,
    soft_nms_max_boxes: int = 1000,
):
    """نسخهٔ Soft-NMS تابع non_max_suppression؛ همان امضا/رفتار نسخهٔ اصلی Ultralytics."""
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"

    if rotated:
        return _ultra_nms.non_max_suppression(
            prediction, conf_thres, iou_thres, classes, agnostic, multi_label,
            labels, max_det, nc, max_time_img, max_nms, max_wh, rotated, end2end, return_idxs,
        )

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs

    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        filt = xc[xi]
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        boxes = x[:, :4] + c

        # ==== تنها تفاوت اصلی نسبت به تابع اصلی Ultralytics ====
        i = soft_nms(
            boxes, scores, iou_thres=iou_thres,
            sigma=soft_nms_sigma, score_thres=soft_nms_score_thres, method=soft_nms_method,
            max_boxes=soft_nms_max_boxes,
        )
        # =========================================================

        i = i[:max_det]

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output


def enable_soft_nms(method: str = "gaussian", sigma: float = 0.5, score_thres: float = 0.001,
                     max_boxes: int = 1000):
    """Soft-NMS را به‌صورت سراسری فعال می‌کند (monkey-patch روی ultralytics.utils.nms.non_max_suppression)."""
    def _patched(*args, **kwargs):
        kwargs.setdefault("soft_nms_method", method)
        kwargs.setdefault("soft_nms_sigma", sigma)
        kwargs.setdefault("soft_nms_score_thres", score_thres)
        kwargs.setdefault("soft_nms_max_boxes", max_boxes)
        return non_max_suppression_soft(*args, **kwargs)

    _ultra_nms.non_max_suppression = _patched
    LOGGER.info(f"✅ Soft-NMS فعال شد (method={method}, sigma={sigma}, score_thres={score_thres}, max_boxes={max_boxes})")


def disable_soft_nms():
    """بازگرداندن NMS استاندارد Ultralytics."""
    import importlib
    importlib.reload(_ultra_nms)
    LOGGER.info("↩️ NMS استاندارد Ultralytics بازگردانده شد.")
