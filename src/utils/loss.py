"""
Loss functions for MASW-YOLO.

Implements Wise-IoU (WIoU) loss for bounding box regression, which provides
a dynamic and wise gradient allocation strategy. WIoU considers the quality
of anchor boxes and adjusts gradients accordingly to improve localization
accuracy for small objects.

Reference:
    "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism"
    https://arxiv.org/abs/2301.10051
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import bbox2dist
# ---------------------------------------------------------------------------
# Wise-IoU helpers
# ---------------------------------------------------------------------------

def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def _iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute full IoU between two sets of boxes in xyxy format."""
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


def _center_distance(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between centers of two boxes in xyxy format."""
    c1 = _box_xyxy_to_cxcywh(boxes1)
    c2 = _box_xyxy_to_cxcywh(boxes2)
    return (c1[..., :2] - c2[..., :2]).pow(2).sum(dim=-1).sqrt()


def _diagonal_length(boxes: torch.Tensor) -> torch.Tensor:
    """Diagonal length of the smallest enclosing box covering both sets."""
    # boxes are xyxy — we need the enclosing box diagonal per pair
    # This is used per-pair, so we only compute from the boxes themselves
    return torch.sqrt(
        (boxes[..., 2] - boxes[..., 0]) ** 2 + (boxes[..., 3] - boxes[..., 1]) ** 2
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wise_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    version: str = "v3",
    beta: float = 1.0,
    delta: float = 0.5,
) -> torch.Tensor:
    """
    Compute Wise-IoU loss between two sets of bounding boxes (xyxy format).

    WIoU v1:  L = 1 - IoU
    WIoU v2:  L = (IoU / (1 - IoU + eps)).detach() * (1 - IoU)   [monotonic FM]
    WIoU v3:  L = r * (1 - IoU)  with r = beta / (delta * alpha ** (delta - beta))
             where alpha = (IoU - IoU.mean()) / (IoU.std() + eps)  [non-monotonic FM]

    Args:
        boxes1 (torch.Tensor): Predicted boxes, shape (N, 4), xyxy format.
        boxes2 (torch.Tensor): Target boxes, shape (N, 4), xyxy format.
        version (str): 'v1', 'v2', or 'v3'. Default: 'v3'.
        beta (float): Focusing parameter for WIoU v2/v3. Default: 1.0.
        delta (float): Gradient allocation parameter for WIoU v3. Default: 0.5.

    Returns:
        torch.Tensor: Wise-IoU loss values, shape (N, 1).
    """
    iou = _iou(boxes1, boxes2)  # (N,)

    # Compute the inner loss L_iou = 1 - IoU  (used as base in all versions)
    loss_iou = 1.0 - iou

    if version == "v1":
        return loss_iou.unsqueeze(-1)

    # --- Compute normalised centre distance (R_wiou) ---
    # R_wiou = exp((center_dist ** 2) / (enclosing_diag ** 2))
    # This is the "wise" part — it up-weights boxes with larger centre offset
    c_dist = _center_distance(boxes1, boxes2)  # (N,)

    # Enclosing box per pair
    enclose_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    enclose_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    enclose_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    enclose_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    enclose_diag = torch.sqrt(
        (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    ).clamp(min=1e-7)

    r_wiou = torch.exp((c_dist ** 2) / (enclose_diag ** 2))  # (N,)

    if version == "v2":
        # WIoU v2: monotonic focusing mechanism
        # L = (IoU / (1 - IoU + eps)).detach() * (1 - IoU)
        # The detach() prevents gradients from flowing through the focusing term
        with torch.no_grad():
            # Use the standard IoU focusing coefficient
            # Paper: L_WIoUv2 = (L_IoU / L_IoU_mean).detach() * L_IoU
            l_iou_mean = loss_iou.mean().detach()
            focusing = (loss_iou / (l_iou_mean + 1e-7)).detach()
        loss = focusing * r_wiou * loss_iou
        return loss.unsqueeze(-1)

    if version == "v3":
        # WIoU v3: non-monotonic focusing mechanism
        # r = beta / (delta * alpha ** (delta - beta))  (with clipping)
        # where alpha = (loss_iou - loss_iou_mean) / (loss_iou_std + eps)
        with torch.no_grad():
            loss_mean = loss_iou.mean().detach()
            loss_std = loss_iou.std().detach() + 1e-7
            alpha = (loss_iou - loss_mean) / loss_std  # normalised deviation
            # Non-monotonic focusing coefficient r
            # Paper: r = exp((alpha ** delta) / (beta * alpha ** (delta - beta)))? 
            # Actually from the paper eq (9): r = beta / (delta * alpha ** (delta - beta))
            # With clipping to avoid extreme values
            r = beta / (delta * alpha.clamp(min=1e-7) ** (delta - beta) + 1e-7)
            r = r.clamp(max=10.0)  # safety clipping

        loss = r * r_wiou * loss_iou
        return loss.unsqueeze(-1)

    raise ValueError(f"Unknown WIoU version: {version}")


class WiseIoULoss(BboxLoss):
    """
    Wise-IoU Loss module.

    Drop-in replacement for ``ultralytics.utils.loss.BboxLoss``.
    Inherits all the infrastructure (DFL loss, etc.) and overrides the
    ``forward`` method to use ``wise_iou`` instead of ``CIoU``.

    Args:
        reg_max (int): Maximum value for the DFL (distribution focal loss).
            Must match the model's ``reg_max`` attribute.
        version (str): WIoU version ('v1', 'v2', 'v3'). Default: 'v3'.
        beta (float): Focusing parameter for WIoU v2/v3. Default: 1.0.
        delta (float): Gradient allocation parameter for WIoU v3. Default: 0.5.
    """

    def __init__(
        self,
        reg_max: int = 16,
        version: str = "v3",
        beta: float = 1.0,
        delta: float = 0.5,
    ):
        super().__init__(reg_max)
        self.version = version
        self.beta = beta
        self.delta = delta

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute WIoU + DFL losses for bounding boxes.

        Args:
            pred_dist (torch.Tensor): Predicted distribution over offsets.
            pred_bboxes (torch.Tensor): Predicted boxes (xyxy format).
            anchor_points (torch.Tensor): Anchor point coordinates.
            target_bboxes (torch.Tensor): Target boxes (xyxy format).
            target_scores (torch.Tensor): Target class scores per anchor.
            target_scores_sum (torch.Tensor): Sum of target scores (for normalisation).
            fg_mask (torch.Tensor): Foreground (positive) mask.
            imgsz (torch.Tensor): Image size (height, width).
            stride (torch.Tensor): Stride values per detection level.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (loss_iou, loss_dfl).
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # ---- Wise-IoU loss ----
        iou = wise_iou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            version=self.version,
            beta=self.beta,
            delta=self.delta,
        )
        loss_iou = (iou * weight).sum() / target_scores_sum

        # ---- DFL loss (unchanged from BboxLoss) ----
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalise ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none")
                .mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl
