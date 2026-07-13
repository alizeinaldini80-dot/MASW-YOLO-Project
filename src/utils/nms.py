"""
src/utils/nms.py

Soft-NMS (Bodla et al., 2017), implementing the score-reset function used in
the MASW-YOLO paper (Eqs. 3-5, Fig. 4):

    Eq. (3) hard-NMS  : si = si            if iou < Nt
                         si = 0             if iou >= Nt

    Eq. (4) linear    : si = si            if iou < Nt
                         si = si*(1-iou)    if iou >= Nt

    Eq. (5) gaussian  : si = si * exp(-iou^2 / sigma)   (no threshold gate)

The paper's final model uses the Gaussian variant (Eq. 5), so `method="gaussian"`
is the default here.

IMPORTANT — scope of this module
---------------------------------
Soft-NMS is a *post-processing* step applied to already-computed detections.
It does not add or remove any layers/weights, so it has zero effect on
Params/M or FLOPs/G. Table 2 of the paper confirms this: the "Soft-NMS only"
row reports the exact same FLOPs/Params as the plain YOLOv8n baseline row
(8.1G / 3.01M). Only P/Recall/mAP change. Do not try to make this file
influence those two metrics — they come entirely from the backbone/neck
architecture (MSCA/AFPN), which lives in separate files and is untouched here.

This module is self-contained: it only monkey-patches
`ultralytics.utils.nms.non_max_suppression` when `enable_soft_nms()` is
called, and can be reverted with `disable_soft_nms()`. It never touches the
model graph, so it is safe to use independently of the MSCA, AFPN, and
WIoU modules.

Usage
-----
    from src.utils.nms import enable_soft_nms, disable_soft_nms

    enable_soft_nms(method="gaussian", sigma=0.5, score_thres=0.001)
    model.val(data="data/dataset.yaml")   # or model.predict(...)
    disable_soft_nms()                    # optional: restore hard-NMS
"""

import time

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

import ultralytics.utils.nms as _ultra_nms


# --------------------------------------------------------------------------- #
# Core Soft-NMS kernel (vectorized: one N x N IoU matrix, no per-box re-query)
# --------------------------------------------------------------------------- #
def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thres: float = 0.5,
    sigma: float = 0.5,
    score_thres: float = 0.001,
    method: str = "gaussian",
    max_boxes: int = 300,
):
    """
    Soft-NMS core, following the Fig. 4 flowchart:
    rank by confidence -> take the top box -> deposit into D -> compute IoU
    of the rest against it -> decay scores -> drop boxes whose score falls
    below the threshold -> repeat until the candidate set is empty.

    Args:
        boxes: (N, 4) xyxy box coordinates.
        scores: (N,) confidence scores.
        iou_thres: IoU threshold Nt. Only used by 'linear' and 'hard'.
        sigma: Gaussian variance parameter (sigma in Eq. 5). Only used by 'gaussian'.
        score_thres: final score cutoff (Si in the paper) applied after decay.
        method: 'gaussian' (paper's Eq. 5, default) | 'linear' (Eq. 4) | 'hard' (Eq. 3).
        max_boxes: pre-filter cap on the number of candidate boxes fed into the
            O(N^2) IoU matrix, to bound memory during validation with many
            low-confidence candidates. Boxes beyond this rank would not have
            survived plain NMS either.

    Returns:
        (keep_idx, keep_scores):
            keep_idx: LongTensor of kept indices, relative to the input boxes/scores.
            keep_scores: FloatTensor of each kept box's *decayed* Soft-NMS score
                (the "si" that must be used for downstream ranking/mAP -- NOT the
                original input score). This is what makes it Soft-NMS rather than
                hard-NMS: a duplicate box is kept but demoted, not deleted.
    """
    n = boxes.shape[0]
    if n == 0:
        return (torch.zeros((0,), dtype=torch.long, device=boxes.device),
                torch.zeros((0,), dtype=boxes.dtype, device=boxes.device))

    device = boxes.device

    # Pre-filter to bound the IoU matrix size.
    if n > max_boxes:
        top = torch.argsort(scores, descending=True)[:max_boxes]
        work_boxes = boxes[top]
        work_scores = scores[top].clone()
    else:
        top = torch.arange(n, device=device)
        work_boxes = boxes
        work_scores = scores.clone()

    m = work_boxes.shape[0]
    iou_mat = box_iou(work_boxes, work_boxes)  # computed once on GPU, m x m

    # The greedy "pick best -> decay -> repeat" step is inherently sequential,
    # so it cannot be vectorized across iterations. Running it on the GPU means
    # each iteration's argmax forces a GPU->CPU sync (a hard stall), which for
    # m in the hundreds/low-thousands (typical at val-time conf=0.001, before
    # NMS) can take seconds per image and blow through Ultralytics' internal
    # per-batch time_limit -- silently truncating detections for the rest of
    # the batch. Moving this part to NumPy on CPU avoids that: a Python loop
    # over ~1000 scalars takes microseconds per step there.
    iou_np = iou_mat.detach().cpu().numpy()
    scores_np = work_scores.detach().cpu().numpy().copy()

    active = np.ones(m, dtype=bool)
    keep_local = []
    keep_scores_local = []

    for _ in range(m):
        remaining = np.where(active)[0]
        if remaining.size == 0:
            break
        i = remaining[np.argmax(scores_np[remaining])]
        if scores_np[i] <= score_thres:
            break

        keep_local.append(int(i))
        # Record the score AT THE MOMENT OF SELECTION -- this already reflects
        # any decay this box received from higher-ranked boxes selected in
        # earlier iterations. This is the value that must be reported downstream,
        # not the box's original pre-decay confidence.
        keep_scores_local.append(float(scores_np[i]))
        active[i] = False
        if not active.any():
            break

        ious = iou_np[i]
        if method == "linear":
            decay = np.where(ious > iou_thres, 1.0 - ious, 1.0)
        elif method == "gaussian":
            decay = np.exp(-(ious * ious) / sigma)
        elif method == "hard":
            decay = np.where(ious > iou_thres, 0.0, 1.0)
        else:
            raise ValueError(f"Invalid Soft-NMS method: {method!r} (use 'gaussian', 'linear', or 'hard')")

        scores_np[active] = scores_np[active] * decay[active]
        active &= scores_np > score_thres

    if not keep_local:
        return (torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=boxes.dtype, device=device))

    keep_local_t = torch.tensor(keep_local, dtype=torch.long, device=device)
    keep_scores_t = torch.tensor(keep_scores_local, dtype=boxes.dtype, device=device)
    return top[keep_local_t], keep_scores_t


# --------------------------------------------------------------------------- #
# Drop-in replacement for ultralytics.utils.nms.non_max_suppression
# --------------------------------------------------------------------------- #
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
    # --- Soft-NMS specific parameters ---
    soft_nms_method: str = "gaussian",
    soft_nms_sigma: float = 0.5,
    soft_nms_score_thres: float = 0.001,
    soft_nms_max_boxes: int = 300,
):
    """Soft-NMS version of `non_max_suppression`; same signature/behavior as
    the Ultralytics original, with the suppression step swapped out."""
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"

    if rotated:
        # Soft-NMS for OBB isn't part of the paper's scope; fall back to stock NMS.
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

        # Class-offset trick: shifts boxes of different classes apart in IoU
        # space so a single (class-agnostic) NMS pass is effectively class-aware.
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        boxes = x[:, :4] + c

        # ==== the only real difference vs. the Ultralytics original ====
        i, decayed_scores = soft_nms(
            boxes, scores, iou_thres=iou_thres,
            sigma=soft_nms_sigma, score_thres=soft_nms_score_thres, method=soft_nms_method,
            max_boxes=soft_nms_max_boxes,
        )
        # =================================================================

        i = i[:max_det]
        decayed_scores = decayed_scores[:max_det]

        # IMPORTANT: use the *decayed* Soft-NMS score, not the box's original
        # confidence. This is what actually makes it Soft-NMS: a duplicate box
        # that overlapped a higher-ranked detection was kept, but demoted, not
        # deleted -- if we reported its original score here it would rank as a
        # high-confidence false positive during mAP evaluation and silently
        # wreck precision (this was a real bug caught via a training run:
        # P/mAP collapsed below the plain-NMS baseline while recall looked
        # normal -- the signature of duplicate boxes ranking too high).
        kept = x[i].clone()
        kept[:, 4] = decayed_scores
        output[xi] = kept
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output


# --------------------------------------------------------------------------- #
# Enable / disable helpers
# --------------------------------------------------------------------------- #
def enable_soft_nms(method: str = "gaussian", sigma: float = 0.5, score_thres: float = 0.001,
                     max_boxes: int = 300):
    """Globally enable Soft-NMS by monkey-patching
    ultralytics.utils.nms.non_max_suppression. Only affects post-processing —
    the model graph (MSCA/AFPN/WIoU) is untouched."""
    def _patched(*args, **kwargs):
        kwargs.setdefault("soft_nms_method", method)
        kwargs.setdefault("soft_nms_sigma", sigma)
        kwargs.setdefault("soft_nms_score_thres", score_thres)
        kwargs.setdefault("soft_nms_max_boxes", max_boxes)
        return non_max_suppression_soft(*args, **kwargs)

    _ultra_nms.non_max_suppression = _patched
    LOGGER.info(f"Soft-NMS enabled (method={method}, sigma={sigma}, score_thres={score_thres}, max_boxes={max_boxes})")


def disable_soft_nms():
    """Restore standard Ultralytics hard-NMS."""
    import importlib
    importlib.reload(_ultra_nms)
    LOGGER.info("Standard Ultralytics NMS restored.")
