"""cv-benchkit: Computer-vision benchmark toolkit."""

from __future__ import annotations

__version__ = "0.2.3"

from cv_benchkit.converters import (
    BBox,
    COCOAnnotation,
    COCODataset,
    coco_to_voc,
    coco_to_yolo,
    cxcywh_to_xyxy,
    normalized_to_xyxy,
    voc_to_coco,
    voc_to_yolo,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_normalized,
    xyxy_to_xywh,
    yolo_to_coco,
    yolo_to_voc,
)
from cv_benchkit.metrics import (
    compute_average_precision,
    compute_ciou,
    compute_confusion_matrix,
    compute_diou,
    compute_giou,
    compute_iou,
    compute_nms,
    compute_precision_recall_curve,
    compute_soft_nms,
)

__all__ = [
    "BBox",
    "COCOAnnotation",
    "COCODataset",
    "__version__",
    "coco_to_voc",
    "coco_to_yolo",
    "compute_average_precision",
    "compute_ciou",
    "compute_confusion_matrix",
    "compute_diou",
    "compute_giou",
    "compute_iou",
    "compute_nms",
    "compute_precision_recall_curve",
    "compute_soft_nms",
    "cxcywh_to_xyxy",
    "normalized_to_xyxy",
    "voc_to_coco",
    "voc_to_yolo",
    "xywh_to_xyxy",
    "xyxy_to_cxcywh",
    "xyxy_to_normalized",
    "xyxy_to_xywh",
    "yolo_to_coco",
    "yolo_to_voc",
]
