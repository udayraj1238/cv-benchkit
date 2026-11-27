"""Evaluation metrics: IoU, GIoU, and confusion matrix computation.

All functions are vectorized with NumPy for efficient batch processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _normalize_boxes(
    boxes: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Coerce boxes to a 2D ``(M, 4)`` xyxy array and validate coordinates.

    An empty 1D input becomes ``(0, 4)``; a non-empty 1D input becomes
    ``(1, 4)``. Any non-empty array without exactly four coordinates on its
    last axis, with a non-finite coordinate (NaN or Inf), or with a minimum
    greater than the corresponding maximum is rejected with a clear error.
    Arrays with more than two dimensions are rejected, even if empty.
    """
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim > 2:
        raise ValueError(f"Boxes array must be 1D or 2D, got {arr.ndim}D array")
    if arr.ndim == 1:
        arr = arr.reshape((0, 4) if arr.size == 0 else (1, -1))
    if arr.size != 0 and arr.shape[-1] != 4:
        raise ValueError("Boxes must have 4 coordinates [x_min, y_min, x_max, y_max]")
    if arr.size != 0 and not np.all(np.isfinite(arr)):
        raise ValueError("Bounding box coordinates must be finite (no NaN or Inf)")
    if arr.size != 0 and np.any(arr[..., :2] > arr[..., 2:]):
        raise ValueError("Box coordinates must satisfy x_min <= x_max and y_min <= y_max")
    return arr


def compute_iou(
    boxes_a: NDArray[np.floating[Any]],
    boxes_b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the pairwise Intersection-over-Union (IoU) matrix.

    Parameters
    ----------
    boxes_a : ndarray, shape ``(M, 4)``
        First set of boxes in xyxy format.
    boxes_b : ndarray, shape ``(N, 4)``
        Second set of boxes in xyxy format.

    Returns
    -------
    ndarray, shape ``(M, N)``
        IoU values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If an input has more than two dimensions, or a box has the wrong number
        of coordinates, non-finite coordinates, or inverted corners.
    """
    boxes_a = _normalize_boxes(boxes_a)
    boxes_b = _normalize_boxes(boxes_b)

    m = boxes_a.shape[0]
    n = boxes_b.shape[0]
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)

    # Intersection
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    # Avoid division by zero
    iou: NDArray[np.floating[Any]] = np.where(union > 0, intersection / union, 0.0)
    return iou


def compute_giou(
    boxes_a: NDArray[np.floating[Any]],
    boxes_b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the pairwise Generalized IoU (GIoU) matrix.

    GIoU = IoU - |C \\ (A U B)| / |C|

    where C is the smallest enclosing box of A and B.

    Parameters
    ----------
    boxes_a : ndarray, shape ``(M, 4)``
        First set of boxes in xyxy format.
    boxes_b : ndarray, shape ``(N, 4)``
        Second set of boxes in xyxy format.

    Returns
    -------
    ndarray, shape ``(M, N)``
        GIoU values in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If an input has more than two dimensions, or a box has the wrong number
        of coordinates, non-finite coordinates, or inverted corners.
    """
    boxes_a = _normalize_boxes(boxes_a)
    boxes_b = _normalize_boxes(boxes_b)

    m = boxes_a.shape[0]
    n = boxes_b.shape[0]
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)

    # Intersection
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    iou = np.where(union > 0, intersection / union, 0.0)

    # Enclosing box
    enc_x1 = np.minimum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    enc_y1 = np.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    enc_x2 = np.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    enc_y2 = np.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou: NDArray[np.floating[Any]] = np.where(
        enc_area > 0,
        iou - (enc_area - union) / enc_area,
        iou,
    )
    return giou


def compute_diou(
    boxes_a: NDArray[np.floating[Any]],
    boxes_b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the pairwise Distance-IoU (DIoU) matrix.

    DIoU = IoU - rho^2 / c^2  (Zheng et al., AAAI 2020)

    where rho is the distance between the two box centers and c is the
    diagonal of the smallest box enclosing both. When ``c^2 <= 0`` (both boxes
    are the same point), DIoU equals IoU.

    Parameters
    ----------
    boxes_a : ndarray, shape ``(M, 4)``
        First set of boxes in xyxy format.
    boxes_b : ndarray, shape ``(N, 4)``
        Second set of boxes in xyxy format.

    Returns
    -------
    ndarray, shape ``(M, N)``
        DIoU values in ``(-1, 1]``.

    Raises
    ------
    ValueError
        If an input has more than two dimensions, or a box has the wrong number
        of coordinates, non-finite coordinates, or inverted corners.
    """
    boxes_a = _normalize_boxes(boxes_a)
    boxes_b = _normalize_boxes(boxes_b)
    m = boxes_a.shape[0]
    n = boxes_b.shape[0]
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)
    iou, penalty = _iou_and_center_penalty(boxes_a, boxes_b)
    diou: NDArray[np.floating[Any]] = iou - penalty
    return diou


def compute_ciou(
    boxes_a: NDArray[np.floating[Any]],
    boxes_b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the pairwise Complete-IoU (CIoU) matrix.

    CIoU = DIoU - alpha * v  (Zheng et al., AAAI 2020)

    with ``v = 4 / pi^2 * (arctan(w_b / h_b) - arctan(w_a / h_a))^2`` measuring
    aspect-ratio disagreement and ``alpha = v / ((1 - IoU) + v)``; ``alpha`` is
    0 when ``v == 0`` or ``(1 - IoU) + v <= 0``. ``arctan(w / h)`` is computed as
    ``arctan2(w, h)`` so zero-height boxes do not divide by zero. ``v`` is
    symmetric, so it does not matter which set is the ground truth.

    Parameters
    ----------
    boxes_a : ndarray, shape ``(M, 4)``
        First set of boxes in xyxy format.
    boxes_b : ndarray, shape ``(N, 4)``
        Second set of boxes in xyxy format.

    Returns
    -------
    ndarray, shape ``(M, N)``
        CIoU values in ``[-1.5, 1]``. DIoU is above -1 and ``alpha * v`` is at
        most 0.5 (reached when IoU is 0), so CIoU can fall below -1 for distant
        boxes with very different aspect ratios.

    Raises
    ------
    ValueError
        If an input has more than two dimensions, or a box has the wrong number
        of coordinates, non-finite coordinates, or inverted corners.
    """
    boxes_a = _normalize_boxes(boxes_a)
    boxes_b = _normalize_boxes(boxes_b)
    m = boxes_a.shape[0]
    n = boxes_b.shape[0]
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)
    iou, penalty = _iou_and_center_penalty(boxes_a, boxes_b)

    angle_a = np.arctan2(boxes_a[:, 2] - boxes_a[:, 0], boxes_a[:, 3] - boxes_a[:, 1])
    angle_b = np.arctan2(boxes_b[:, 2] - boxes_b[:, 0], boxes_b[:, 3] - boxes_b[:, 1])
    v = (4.0 / np.pi**2) * (angle_b[None, :] - angle_a[:, None]) ** 2
    denominator = (1.0 - iou) + v
    safe_denominator = np.where(denominator > 0, denominator, 1.0)
    alpha = np.where((v > 0) & (denominator > 0), v / safe_denominator, 0.0)

    ciou: NDArray[np.floating[Any]] = iou - penalty - alpha * v
    return ciou


def _iou_and_center_penalty(
    boxes_a: NDArray[np.floating[Any]],
    boxes_b: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Return the pairwise IoU and the DIoU center-distance term ``rho^2 / c^2``.

    Both inputs must already be normalized and non-empty. The penalty is 0
    where the enclosing diagonal ``c^2`` is not positive.
    """
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersection = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    safe_union = np.where(union > 0, union, 1.0)
    iou = np.where(union > 0, intersection / safe_union, 0.0)

    center_ax = (boxes_a[:, 0] + boxes_a[:, 2]) / 2.0
    center_ay = (boxes_a[:, 1] + boxes_a[:, 3]) / 2.0
    center_bx = (boxes_b[:, 0] + boxes_b[:, 2]) / 2.0
    center_by = (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0
    rho2 = (center_ax[:, None] - center_bx[None, :]) ** 2 + (
        center_ay[:, None] - center_by[None, :]
    ) ** 2

    enc_w = np.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2]) - np.minimum(
        boxes_a[:, None, 0], boxes_b[None, :, 0]
    )
    enc_h = np.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3]) - np.minimum(
        boxes_a[:, None, 1], boxes_b[None, :, 1]
    )
    c2 = enc_w**2 + enc_h**2
    safe_c2 = np.where(c2 > 0, c2, 1.0)
    penalty = np.where(c2 > 0, rho2 / safe_c2, 0.0)
    return iou, penalty


def compute_nms(
    boxes: NDArray[np.floating[Any]],
    scores: NDArray[np.floating[Any]],
    iou_threshold: float = 0.5,
    class_ids: NDArray[np.integer[Any]] | None = None,
) -> NDArray[np.int64]:
    """Perform greedy non-maximum suppression (NMS) on bounding boxes.

    Boxes are visited in order of descending score. Each visited box that has
    not been suppressed is kept, and every remaining box whose IoU with it
    is greater than ``iou_threshold`` is suppressed. Boxes with equal scores
    keep their input order (stable sort), so the result is deterministic.

    IoU is computed with :func:`compute_iou`, one kept box against the
    remaining candidates at a time, so memory stays ``O(N)`` instead of
    building the full ``(N, N)`` IoU matrix.

    Parameters
    ----------
    boxes : ndarray, shape ``(N, 4)``
        Bounding boxes in ``[x1, y1, x2, y2]`` format.
    scores : ndarray, shape ``(N,)``
        Confidence scores for each box. Must be finite.
    iou_threshold : float, default 0.5
        IoU threshold for suppression (must be in ``[0.0, 1.0]``). A box is
        suppressed only when its IoU is strictly greater than this value, so
        ``1.0`` keeps every box.
    class_ids : ndarray, shape ``(N,)``, optional
        Integer class IDs. If provided, NMS is performed independently per
        class: a box only suppresses boxes of the same class.

    Returns
    -------
    ndarray, shape ``(K,)``, dtype int64
        Indices of the retained boxes, sorted by descending score.

    Raises
    ------
    ValueError
        If ``boxes`` is not ``(N, 4)``, ``scores`` is not a 1D array of
        length ``N`` or has a non-finite value, ``iou_threshold`` is outside
        ``[0.0, 1.0]``, ``class_ids`` is not a 1D array of length ``N``, or a
        box is invalid (non-finite or inverted coordinates).

    Examples
    --------
    >>> boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=float)
    >>> scores = np.array([0.9, 0.8, 0.7])
    >>> compute_nms(boxes, scores, iou_threshold=0.5)
    array([0, 2])
    >>> compute_nms(boxes, scores, iou_threshold=0.5, class_ids=np.array([0, 1, 0]))
    array([0, 1, 2])
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}")

    boxes_arr = np.asarray(boxes, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    if boxes_arr.shape == (0,) and scores_arr.size == 0:
        boxes_arr = boxes_arr.reshape(0, 4)
    if boxes_arr.ndim != 2 or boxes_arr.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {boxes_arr.shape}")
    normalized_boxes = _normalize_boxes(boxes_arr)
    n = normalized_boxes.shape[0]
    if scores_arr.ndim != 1 or scores_arr.shape[0] != n:
        raise ValueError(f"scores must be a 1D array of length {n}, got shape {scores_arr.shape}")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("scores must be finite (no NaN or Inf)")

    classes: NDArray[Any] | None = None
    if class_ids is not None:
        classes = np.asarray(class_ids)
        if classes.ndim != 1 or classes.shape[0] != n:
            raise ValueError(
                f"class_ids must be a 1D array of length {n}, got shape {classes.shape}"
            )

    if n == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(-scores_arr, kind="stable")
    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        if rest.size == 0:
            break
        overlaps = (
            compute_iou(normalized_boxes[best : best + 1], normalized_boxes[rest])[0]
            > iou_threshold
        )
        if classes is not None:
            overlaps &= classes[rest] == classes[best]
        order = rest[~overlaps]
    return np.asarray(keep, dtype=np.int64)


def compute_soft_nms(
    boxes: NDArray[np.floating[Any]],
    scores: NDArray[np.floating[Any]],
    method: str = "linear",
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Soft-NMS to bounding boxes and confidence scores.

    Soft-NMS (Bodla et al., ICCV 2017, "Soft-NMS -- Improving Object Detection
    With One Line of Code") follows the greedy loop of :func:`compute_nms`, but
    instead of removing every box that overlaps the selected one it lowers the
    box's score as a function of their IoU:

    1. Select the remaining box with the highest current score.
    2. Multiply the score of every other remaining box by a decay weight
       computed from its IoU with the selected box.
    3. Drop the remaining boxes whose score is now below ``score_threshold``,
       and repeat until no box remains.

    The decay weights are

    * ``"linear"``: ``1 - iou`` when ``iou >= iou_threshold``, else ``1``;
    * ``"gaussian"``: ``exp(-(iou ** 2) / sigma)``, for every IoU.

    Scores only ever decrease, so the boxes come out in the order they were
    selected, which is descending updated score. Boxes with equal scores are
    selected in input order, so the result is deterministic. A box whose
    initial score is already below ``score_threshold`` is never kept.

    IoU is computed with :func:`compute_iou`, one selected box against the
    remaining boxes at a time, so memory stays ``O(N)``.

    Parameters
    ----------
    boxes : ndarray, shape ``(N, 4)``
        Bounding boxes in ``[x1, y1, x2, y2]`` format.
    scores : ndarray, shape ``(N,)``
        Confidence scores for each box. Must be finite.
    method : {"linear", "gaussian"}, default "linear"
        Score decay rule (see above).
    iou_threshold : float, default 0.5
        IoU at and above which the linear rule decays a score. Must be in
        ``[0.0, 1.0]``; it is validated but not used by the Gaussian rule.
    sigma : float, default 0.5
        Spread of the Gaussian rule. Must be ``> 0``.
    score_threshold : float, default 0.001
        Boxes whose (decayed) score falls below this value are discarded.

    Returns
    -------
    kept_boxes : ndarray, shape ``(K, 4)``, dtype float64
        The retained boxes, sorted by updated score descending.
    kept_scores : ndarray, shape ``(K,)``, dtype float64
        The updated confidence scores of ``kept_boxes``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"linear"`` or ``"gaussian"``, ``sigma <= 0``,
        ``iou_threshold`` is outside ``[0.0, 1.0]``, ``boxes`` is not
        ``(N, 4)``, ``scores`` is not a 1D array of length ``N`` or has a
        non-finite value, or a box is invalid (non-finite or inverted
        coordinates).

    Examples
    --------
    >>> boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 8], [20, 20, 30, 30]], dtype=float)
    >>> scores = np.array([0.9, 0.8, 0.7])
    >>> kept_boxes, kept_scores = compute_soft_nms(boxes, scores)
    >>> kept_scores.round(2)
    array([0.9 , 0.7 , 0.16])
    >>> kept_boxes[2]
    array([ 0.,  0., 10.,  8.])
    >>> compute_soft_nms(boxes, scores, method="gaussian")[1].round(3)
    array([0.9  , 0.7  , 0.222])
    """
    if method not in ("linear", "gaussian"):
        raise ValueError(f"Unknown method {method!r}, expected 'linear' or 'gaussian'")
    if not sigma > 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}")

    boxes_arr = np.asarray(boxes, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    if boxes_arr.shape == (0,) and scores_arr.size == 0:
        boxes_arr = boxes_arr.reshape(0, 4)
    if boxes_arr.ndim != 2 or boxes_arr.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {boxes_arr.shape}")
    normalized_boxes = _normalize_boxes(boxes_arr)
    n = normalized_boxes.shape[0]
    if scores_arr.ndim != 1 or scores_arr.shape[0] != n:
        raise ValueError(f"scores must be a 1D array of length {n}, got shape {scores_arr.shape}")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("scores must be finite (no NaN or Inf)")

    current = scores_arr.copy()
    remaining = np.flatnonzero(current >= score_threshold)
    keep: list[int] = []
    while remaining.size > 0:
        # argmax returns the first maximum, so ties go to the lowest input index.
        best = int(remaining[np.argmax(current[remaining])])
        keep.append(best)
        remaining = remaining[remaining != best]
        if remaining.size == 0:
            break
        iou = compute_iou(normalized_boxes[best : best + 1], normalized_boxes[remaining])[0]
        if method == "linear":
            weight = np.where(iou >= iou_threshold, 1.0 - iou, 1.0)
        else:
            weight = np.exp(-(iou**2) / sigma)
        current[remaining] *= weight
        remaining = remaining[current[remaining] >= score_threshold]

    index = np.asarray(keep, dtype=np.int64)
    kept_boxes = np.asarray(normalized_boxes[index], dtype=np.float64).reshape(-1, 4)
    kept_scores = np.asarray(current[index], dtype=np.float64)
    return kept_boxes, kept_scores


def compute_confusion_matrix(
    gt_labels: NDArray[np.integer[Any]],
    pred_labels: NDArray[np.integer[Any]],
    gt_boxes: NDArray[np.floating[Any]],
    pred_boxes: NDArray[np.floating[Any]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> NDArray[np.int64]:
    """Compute an object-detection confusion matrix.

    Predictions are matched to ground-truth boxes using IoU. The matrix has
    shape ``(num_classes + 1, num_classes + 1)`` where the last row/column
    represents the "background" class (unmatched GT = FN, unmatched pred = FP).

    Parameters
    ----------
    gt_labels : ndarray, shape ``(G,)``
        Ground-truth class labels (0-indexed).
    pred_labels : ndarray, shape ``(P,)``
        Predicted class labels (0-indexed).
    gt_boxes : ndarray, shape ``(G, 4)``
        Ground-truth boxes in xyxy format.
    pred_boxes : ndarray, shape ``(P, 4)``
        Predicted boxes in xyxy format.
    num_classes : int
        Number of object classes (not counting background).
    iou_threshold : float
        IoU threshold for a match to count as a true positive. Must be in ``[0, 1]``.

    Returns
    -------
    ndarray, shape ``(num_classes + 1, num_classes + 1)``
        Confusion matrix where ``[i, j]`` = count of GT class *i* predicted as
        class *j*. The last index is "background".

    Raises
    ------
    ValueError
        If ``iou_threshold`` is outside ``[0, 1]``, ``gt_labels`` or
        ``pred_labels`` has more than one dimension, ``num_classes`` is not
        positive, any class label is outside ``[0, num_classes - 1]``, or the
        label and box counts do not match.
    """
    if not (0.0 <= iou_threshold <= 1.0):
        msg = f"iou_threshold must be between 0.0 and 1.0, got {iou_threshold}"
        raise ValueError(msg)

    gt_labels = np.asarray(gt_labels, dtype=np.int64)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)

    if gt_labels.ndim > 1 or pred_labels.ndim > 1:
        raise ValueError("gt_labels and pred_labels must be 1D arrays")

    if num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")

    if (gt_labels.size > 0 and (np.any(gt_labels < 0) or np.any(gt_labels >= num_classes))) or (
        pred_labels.size > 0 and (np.any(pred_labels < 0) or np.any(pred_labels >= num_classes))
    ):
        raise ValueError(f"Class labels must be in range [0, {num_classes - 1}]")

    if len(gt_labels) != len(gt_boxes):
        raise ValueError("Number of ground-truth labels must match number of ground-truth boxes")
    if len(pred_labels) != len(pred_boxes):
        raise ValueError("Number of predicted labels must match number of predicted boxes")

    bg = num_classes  # background index
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    num_gt = gt_labels.shape[0]
    num_pred = pred_labels.shape[0]

    if num_gt == 0 and num_pred == 0:
        return cm

    if num_gt == 0:
        # All predictions are false positives
        for pl in pred_labels:
            cm[bg, int(pl)] += 1
        return cm

    if num_pred == 0:
        # All ground truths are false negatives
        for gl in gt_labels:
            cm[int(gl), bg] += 1
        return cm

    iou_matrix = compute_iou(gt_boxes, pred_boxes)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    # Greedy matching: for each pred, find best IoU GT
    for pred_idx in range(num_pred):
        ious = iou_matrix[:, pred_idx]
        best_gt_idx = int(np.argmax(ious))
        if ious[best_gt_idx] >= iou_threshold and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            gt_cls = int(gt_labels[best_gt_idx])
            pred_cls = int(pred_labels[pred_idx])
            cm[gt_cls, pred_cls] += 1
        else:
            # False positive
            pred_cls = int(pred_labels[pred_idx])
            cm[bg, pred_cls] += 1

    # Unmatched GT → false negatives
    for gt_idx in range(num_gt):
        if gt_idx not in matched_gt:
            gt_cls = int(gt_labels[gt_idx])
            cm[gt_cls, bg] += 1

    return cm


def compute_precision_recall_curve(
    scores: NDArray[np.floating[Any]],
    matches: NDArray[np.bool_],
    num_positives: int,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Compute precision and recall values ordered by descending confidence.

    Parameters
    ----------
    scores : ndarray, shape ``(N,)``
        1D array of detection confidence scores.
    matches : ndarray, shape ``(N,)``
        1D boolean array where ``True`` indicates a true positive and
        ``False`` indicates a false positive.
    num_positives : int
        Total number of ground-truth positives (must be ``>= 0``).

    Returns
    -------
    tuple of (ndarray, ndarray)
        ``(precision, recall)`` 1D float64 arrays sorted by descending
        ``scores``. Returns ``(np.empty(0), np.empty(0))`` when
        ``num_positives == 0`` or ``scores`` is empty.

    Raises
    ------
    ValueError
        If ``scores`` and ``matches`` are not 1D arrays of equal length, or
        if ``num_positives < 0``.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    matches_arr = np.asarray(matches, dtype=np.bool_)

    if scores_arr.ndim != 1 or matches_arr.ndim != 1:
        raise ValueError("scores and matches must be 1D arrays")
    if len(scores_arr) != len(matches_arr):
        raise ValueError(
            "scores and matches must have the same length, "
            f"got {len(scores_arr)} and {len(matches_arr)}"
        )
    if num_positives < 0:
        raise ValueError(f"num_positives must be non-negative, got {num_positives}")

    if num_positives == 0 or scores_arr.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    order = np.argsort(-scores_arr, kind="mergesort")
    sorted_matches = matches_arr[order]

    cumulative_tp = np.cumsum(sorted_matches, dtype=np.float64)
    cumulative_fp = np.cumsum(~sorted_matches, dtype=np.float64)

    recall: NDArray[np.floating[Any]] = cumulative_tp / float(num_positives)
    denom = cumulative_tp + cumulative_fp
    precision: NDArray[np.floating[Any]] = np.where(denom > 0, cumulative_tp / denom, 0.0)
    return precision, recall


def compute_average_precision(
    precision: NDArray[np.floating[Any]],
    recall: NDArray[np.floating[Any]],
    method: str = "all_points",
) -> float:
    """Compute Average Precision (AP) from precision and recall arrays.

    Parameters
    ----------
    precision : ndarray, shape ``(N,)``
        1D array of precision values in ``[0.0, 1.0]``.
    recall : ndarray, shape ``(N,)``
        1D array of recall values in ``[0.0, 1.0]``.
    method : {"all_points", "interp_11"}, default="all_points"
        Interpolation method:
        - ``"all_points"``: COCO / PASCAL VOC 2010+ all-point interpolation.
        - ``"interp_11"``: PASCAL VOC 2007 11-point interpolation.

    Returns
    -------
    float
        Average Precision score in ``[0.0, 1.0]``. Returns ``0.0`` for empty
        inputs.

    Raises
    ------
    ValueError
        If ``precision`` and ``recall`` are not 1D arrays of equal length,
        contain values outside ``[0.0, 1.0]`` (or non-finite), or if
        ``method`` is unsupported.
    """
    if method not in ("all_points", "interp_11"):
        raise ValueError(f"Unsupported AP method '{method}'. Expected 'all_points' or 'interp_11'.")

    prec = np.asarray(precision, dtype=np.float64)
    rec = np.asarray(recall, dtype=np.float64)

    if prec.ndim != 1 or rec.ndim != 1:
        raise ValueError("precision and recall must be 1D arrays")
    if prec.shape != rec.shape:
        raise ValueError(
            f"precision and recall must have the same shape, got {prec.shape} and {rec.shape}"
        )
    if prec.size == 0:
        return 0.0

    if not (np.all(np.isfinite(prec)) and np.all(np.isfinite(rec))):
        raise ValueError("precision and recall must be finite")
    if np.any((prec < 0.0) | (prec > 1.0)) or np.any((rec < 0.0) | (rec > 1.0)):
        raise ValueError("precision and recall values must be in [0.0, 1.0]")

    if method == "interp_11":
        thresholds = np.linspace(0.0, 1.0, 11)
        ap_sum = 0.0
        for thr in thresholds:
            mask = rec >= thr
            ap_sum += float(np.max(prec[mask])) if np.any(mask) else 0.0
        return ap_sum / 11.0

    # "all_points": sentinel (recall=0, precision=1) and (recall=1, precision=0)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for idx in range(mpre.size - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])

    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))
