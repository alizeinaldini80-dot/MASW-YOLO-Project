"""
src/utils/nms.py

پیاده‌سازی Soft-NMS (Bodla et al., 2017) به‌عنوان جایگزین NMS سخت
(hard-NMS) استاندارد، طبق بخش مربوطه در مقاله MASW-YOLO.

تفاوت با NMS معمولی:
    NMS سخت باکس‌های هم‌پوشان با IoU بالاتر از آستانه را کاملاً حذف
    می‌کند (امتیاز = صفر). این کار در صحنه‌های شلوغ مثل VisDrone
    (اشیای کوچک و چگال مثل عابر پیاده/موتور) باعث حذف اشتباهِ
    تشخیص‌های درست هم‌پوشان می‌شود.
    Soft-NMS به‌جای حذف کامل، امتیاز باکس‌های هم‌پوشان را متناسب با
    میزان IoU کاهش می‌دهد (decay) و فقط در انتها باکس‌هایی که امتیازشان
    از آستانه پایین‌تر رفته حذف می‌شوند.

این ماژول دو بخش دارد:
    1) soft_nms(...)            : هستهٔ الگوریتم (وکتوریزه، روی GPU/CPU)
    2) enable_soft_nms(...)     : monkey-patch سراسری که تابع
       ultralytics.utils.nms.non_max_suppression را با نسخه‌ای که در
       داخل از soft_nms استفاده می‌کند جایگزین می‌کند.

نحوهٔ استفاده (برای آزمایش مستقل Soft-NMS روی baseline، بدون MSCA/AFPN):
    from src.utils.nms import enable_soft_nms
    enable_soft_nms(method="gaussian", sigma=0.5, score_thres=0.001)
    # از این‌جا به بعد، هر train/val/predict که ultralytics انجام بدهد
    # (چه در حین آموزش، چه validate جداگانه) به‌طور خودکار از Soft-NMS
    # به‌جای NMS سخت استفاده می‌کند.

⚠️ نکتهٔ نسخه: پیاده‌سازی non_max_suppression_soft در این فایل، عیناً بر
اساس ساختار تابع اصلی ultralytics.utils.nms.non_max_suppression (نسخهٔ
نصب‌شده هنگام نوشتن این کد) است. اگر بعد از آپدیت نسخه در کولب خطای
غیرمنتظره گرفتید، این را اجرا و با این فایل مقایسه کنید:
    import inspect, ultralytics.utils.nms as n
    print(inspect.getsource(n.non_max_suppression))
"""

import time

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

# مرجع تابع اصلی (برای مسیر rotated/OBB که Soft-NMS برایش پیاده نشده)
import ultralytics.utils.nms as _ultra_nms


def soft_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float = 0.5,
             sigma: float = 0.5, score_thres: float = 0.001, method: str = "gaussian"):
    """
    هستهٔ الگوریتم Soft-NMS.

    Args:
        boxes (Tensor[N,4]): مختصات xyxy
        scores (Tensor[N]): امتیاز اطمینان هر باکس
        iou_thres (float): آستانهٔ IoU (فقط برای method='linear' یا 'hard' استفاده می‌شود)
        sigma (float): پارامتر واریانس گاووسی (فقط برای method='gaussian')
        score_thres (float): آستانهٔ نهایی حذف باکس‌های امتیاز-کاهش‌یافته
        method (str): 'gaussian' | 'linear' | 'hard' (hard = معادل NMS معمولی)

    Returns:
        Tensor[K]: اندیس باکس‌های نگه‌داشته‌شده (در dtype long)، به ترتیب انتخاب (نزولی بر اساس امتیاز)
    """
    if boxes.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    device = boxes.device
    scores = scores.clone()
    remaining = torch.arange(scores.shape[0], device=device)
    keep = []

    while remaining.numel() > 0:
        # انتخاب بالاترین امتیاز باقی‌مانده
        local_max = torch.argmax(scores[remaining])
        i = remaining[local_max]
        keep.append(i.item())

        remaining = remaining[remaining != i]
        if remaining.numel() == 0:
            break

        ious = box_iou(boxes[i].unsqueeze(0), boxes[remaining]).squeeze(0)

        if method == "linear":
            decay = torch.where(ious > iou_thres, 1.0 - ious, torch.ones_like(ious))
        elif method == "gaussian":
            decay = torch.exp(-(ious * ious) / sigma)
        elif method == "hard":
            decay = torch.where(ious > iou_thres, torch.zeros_like(ious), torch.ones_like(ious))
        else:
            raise ValueError(f"روش نامعتبر برای Soft-NMS: {method}")

        scores[remaining] = scores[remaining] * decay

        valid = scores[remaining] > score_thres
        remaining = remaining[valid]

    return torch.tensor(keep, dtype=torch.long, device=device)


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
):
    """
    نسخهٔ Soft-NMS تابع non_max_suppression؛ همان امضا و رفتار نسخهٔ اصلی
    Ultralytics را دارد (تا با val.py / predict.py سازگار باشد)، فقط
    مرحلهٔ سرکوب سخت (torchvision.ops.nms / TorchNMS.nms) با soft_nms
    جایگزین شده. برای OBB/rotated که Soft-NMS پیاده نشده، مسیر اصلی
    ultralytics به‌عنوان fallback فراخوانی می‌شود.
    """
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"

    if rotated:
        # Soft-NMS برای OBB پیاده نشده؛ به پیاده‌سازی اصلی واگذار می‌شود
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


def enable_soft_nms(method: str = "gaussian", sigma: float = 0.5, score_thres: float = 0.001):
    """
    Soft-NMS را به‌صورت سراسری فعال می‌کند: تابع
    ultralytics.utils.nms.non_max_suppression را monkey-patch می‌کند تا
    val.py و predict.py (که هر دو `nms.non_max_suppression(...)` را در
    زمان فراخوانی از ماژول resolve می‌کنند) به‌طور خودکار از این تابع
    استفاده کنند. تنظیمات method/sigma/score_thres به‌صورت closure بسته
    می‌شوند تا امضای تابع دقیقاً با نسخهٔ اصلی سازگار بماند.
    """
    def _patched(*args, **kwargs):
        kwargs.setdefault("soft_nms_method", method)
        kwargs.setdefault("soft_nms_sigma", sigma)
        kwargs.setdefault("soft_nms_score_thres", score_thres)
        return non_max_suppression_soft(*args, **kwargs)

    _ultra_nms.non_max_suppression = _patched
    LOGGER.info(f"✅ Soft-NMS فعال شد (method={method}, sigma={sigma}, score_thres={score_thres})")


def disable_soft_nms():
    """بازگرداندن NMS استاندارد Ultralytics (در صورت نیاز به مقایسه در همان اجرا)."""
    import importlib
    importlib.reload(_ultra_nms)
    LOGGER.info("↩️ NMS استاندارد Ultralytics بازگردانده شد.")
