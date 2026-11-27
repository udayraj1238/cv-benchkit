"""Unit tests for compute_soft_nms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cv_benchkit import compute_iou, compute_nms, compute_soft_nms


def _reference_soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    method: str,
    iou_threshold: float,
    sigma: float,
    score_threshold: float,
) -> tuple[list[int], list[float]]:
    """Algorithm 1 of Bodla et al. (2017), written out with plain Python loops."""
    iou = compute_iou(boxes, boxes)
    current = {i: float(s) for i, s in enumerate(scores) if s >= score_threshold}
    keep: list[int] = []
    kept_scores: list[float] = []
    while current:
        best = min(current, key=lambda i: (-current[i], i))
        keep.append(best)
        kept_scores.append(current.pop(best))
        for i in list(current):
            overlap = float(iou[best, i])
            if method == "linear":
                weight = 1.0 - overlap if overlap >= iou_threshold else 1.0
            else:
                weight = math.exp(-(overlap**2) / sigma)
            current[i] *= weight
            if current[i] < score_threshold:
                del current[i]
    return keep, kept_scores


IDENTICAL = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
DISJOINT = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=float)
OVERLAPPING = np.array([[0, 0, 10, 10], [0, 0, 10, 8], [20, 20, 30, 30]], dtype=float)


class TestIdenticalBoxes:
    def test_linear_decay_removes_exact_duplicates(self) -> None:
        # IoU 1 gives a linear weight of 0, below any positive score_threshold.
        kept_boxes, kept_scores = compute_soft_nms(IDENTICAL, np.array([0.9, 0.8, 0.7]))
        np.testing.assert_array_equal(kept_boxes, IDENTICAL[:1])
        np.testing.assert_allclose(kept_scores, [0.9])

    def test_gaussian_decay_lowers_scores_without_zeroing_them(self) -> None:
        kept_boxes, kept_scores = compute_soft_nms(
            IDENTICAL, np.array([0.9, 0.8, 0.7]), method="gaussian", sigma=0.5
        )
        decay = math.exp(-1.0 / 0.5)
        # The second box is decayed once (by the first); the third twice.
        np.testing.assert_allclose(kept_scores, [0.9, 0.8 * decay, 0.7 * decay**2])
        assert kept_boxes.shape == (3, 4)

    def test_gaussian_decay_below_score_threshold_drops_the_box(self) -> None:
        _, kept_scores = compute_soft_nms(
            IDENTICAL, np.array([0.9, 0.8, 0.7]), method="gaussian", score_threshold=0.05
        )
        # 0.7 * exp(-2) ** 2 ~= 0.013 falls below 0.05; 0.8 * exp(-2) ~= 0.108 does not.
        np.testing.assert_allclose(kept_scores, [0.9, 0.8 * math.exp(-2.0)])


class TestDisjointBoxes:
    @pytest.mark.parametrize("method", ["linear", "gaussian"])
    def test_scores_are_unchanged(self, method: str) -> None:
        scores = np.array([0.5, 0.9, 0.7])
        kept_boxes, kept_scores = compute_soft_nms(DISJOINT, scores, method=method)
        np.testing.assert_array_equal(kept_scores, [0.9, 0.7, 0.5])
        np.testing.assert_array_equal(kept_boxes, DISJOINT[[1, 2, 0]])


class TestDecayRules:
    def test_linear_decays_at_and_above_the_threshold_only(self) -> None:
        # IoU of the second box with the first is exactly 0.8.
        assert compute_iou(OVERLAPPING[:1], OVERLAPPING[1:2])[0, 0] == pytest.approx(0.8)
        scores = np.array([0.9, 0.8, 0.7])
        _, at = compute_soft_nms(OVERLAPPING, scores, iou_threshold=0.8)
        _, above = compute_soft_nms(OVERLAPPING, scores, iou_threshold=0.81)
        np.testing.assert_allclose(at, [0.9, 0.7, 0.8 * (1 - 0.8)])
        np.testing.assert_allclose(above, [0.9, 0.8, 0.7])

    def test_gaussian_ignores_iou_threshold(self) -> None:
        scores = np.array([0.9, 0.8, 0.7])
        low = compute_soft_nms(OVERLAPPING, scores, method="gaussian", iou_threshold=0.0)
        high = compute_soft_nms(OVERLAPPING, scores, method="gaussian", iou_threshold=1.0)
        np.testing.assert_array_equal(low[1], high[1])
        np.testing.assert_allclose(low[1], [0.9, 0.7, 0.8 * math.exp(-0.64 / 0.5)])

    def test_smaller_sigma_decays_more(self) -> None:
        scores = np.array([0.9, 0.8, 0.7])
        wide = compute_soft_nms(OVERLAPPING, scores, method="gaussian", sigma=2.0)[1]
        narrow = compute_soft_nms(OVERLAPPING, scores, method="gaussian", sigma=0.1)[1]
        assert narrow[-1] < wide[-1]

    def test_a_decayed_box_can_be_overtaken(self) -> None:
        # After decay, box 1 (0.8 -> 0.16) ranks below the untouched box 2 (0.7).
        kept_boxes, kept_scores = compute_soft_nms(OVERLAPPING, np.array([0.9, 0.8, 0.7]))
        np.testing.assert_array_equal(kept_boxes, OVERLAPPING[[0, 2, 1]])
        np.testing.assert_allclose(kept_scores, [0.9, 0.7, 0.16])

    def test_matches_the_reference_on_random_inputs(self) -> None:
        rng = np.random.default_rng(0)
        for trial in range(200):
            n = int(rng.integers(0, 25))
            xy = rng.uniform(0, 50, size=(n, 2))
            wh = rng.uniform(1, 30, size=(n, 2))
            boxes = np.hstack([xy, xy + wh])
            scores = rng.uniform(0, 1, size=n)
            if trial % 4 == 0:
                scores = np.round(scores, 1)  # force ties
            method = "linear" if trial % 2 else "gaussian"
            iou_threshold = float(rng.uniform(0, 1))
            sigma = float(rng.uniform(0.05, 2))
            score_threshold = float(rng.choice([0.0, 0.001, 0.1, 0.3]))
            kept_boxes, kept_scores = compute_soft_nms(
                boxes, scores, method, iou_threshold, sigma, score_threshold
            )
            ref_keep, ref_scores = _reference_soft_nms(
                boxes, scores, method, iou_threshold, sigma, score_threshold
            )
            np.testing.assert_array_equal(kept_boxes, boxes[ref_keep].reshape(-1, 4))
            np.testing.assert_allclose(kept_scores, ref_scores, rtol=1e-12, atol=0)


class TestOrdering:
    def test_output_is_sorted_by_updated_score_descending(self) -> None:
        rng = np.random.default_rng(1)
        xy = rng.uniform(0, 40, size=(60, 2))
        boxes = np.hstack([xy, xy + rng.uniform(5, 20, size=(60, 2))])
        scores = rng.uniform(0, 1, size=60)
        for method in ("linear", "gaussian"):
            _, kept_scores = compute_soft_nms(boxes, scores, method=method, iou_threshold=0.3)
            assert np.all(np.diff(kept_scores) <= 0)

    def test_equal_scores_keep_input_order(self) -> None:
        kept_boxes, kept_scores = compute_soft_nms(DISJOINT[[2, 0, 1]], np.array([0.5, 0.5, 0.5]))
        np.testing.assert_array_equal(kept_boxes, DISJOINT[[2, 0, 1]])
        np.testing.assert_array_equal(kept_scores, [0.5, 0.5, 0.5])

    def test_is_deterministic(self) -> None:
        scores = np.array([0.6, 0.6, 0.6])
        first = compute_soft_nms(IDENTICAL, scores, method="gaussian")
        second = compute_soft_nms(IDENTICAL, scores, method="gaussian")
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class TestScoreThreshold:
    def test_boxes_starting_below_the_threshold_are_dropped(self) -> None:
        kept_boxes, kept_scores = compute_soft_nms(DISJOINT, np.array([0.9, 0.0005, 0.5]))
        np.testing.assert_array_equal(kept_boxes, DISJOINT[[0, 2]])
        np.testing.assert_array_equal(kept_scores, [0.9, 0.5])

    def test_no_box_survives(self) -> None:
        kept_boxes, kept_scores = compute_soft_nms(
            DISJOINT, np.array([0.1, 0.2, 0.3]), score_threshold=0.5
        )
        assert kept_boxes.shape == (0, 4)
        assert kept_scores.shape == (0,)

    def test_linear_soft_nms_keeps_every_box_hard_nms_keeps(self) -> None:
        rng = np.random.default_rng(3)
        xy = rng.uniform(0, 40, size=(40, 2))
        boxes = np.hstack([xy, xy + rng.uniform(5, 20, size=(40, 2))])
        scores = rng.uniform(0.5, 1, size=40)
        hard = {tuple(b) for b in boxes[compute_nms(boxes, scores, iou_threshold=0.5)]}
        soft = {tuple(b) for b in compute_soft_nms(boxes, scores, score_threshold=0.0)[0]}
        assert hard <= soft


class TestEmptyInput:
    @pytest.mark.parametrize("boxes", [np.empty((0, 4)), np.array([])])
    @pytest.mark.parametrize("method", ["linear", "gaussian"])
    def test_returns_empty_arrays(self, boxes: np.ndarray, method: str) -> None:
        kept_boxes, kept_scores = compute_soft_nms(boxes, np.array([]), method=method)
        assert kept_boxes.shape == (0, 4)
        assert kept_scores.shape == (0,)
        assert kept_boxes.dtype == np.float64
        assert kept_scores.dtype == np.float64


class TestValidation:
    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(
            ValueError, match="Unknown method 'hard', expected 'linear' or 'gaussian'"
        ):
            compute_soft_nms(DISJOINT, np.ones(3), method="hard")

    @pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan")])
    def test_rejects_non_positive_sigma(self, sigma: float) -> None:
        with pytest.raises(ValueError, match="sigma must be > 0"):
            compute_soft_nms(DISJOINT, np.ones(3), method="gaussian", sigma=sigma)

    @pytest.mark.parametrize("iou_threshold", [-0.1, 1.1, float("nan")])
    def test_rejects_iou_threshold_outside_unit_interval(self, iou_threshold: float) -> None:
        with pytest.raises(ValueError, match="iou_threshold must be in"):
            compute_soft_nms(DISJOINT, np.ones(3), iou_threshold=iou_threshold)

    @pytest.mark.parametrize(
        "boxes", [np.zeros((3, 5)), np.zeros(4), np.zeros((2, 3, 4))], ids=["5-cols", "1d", "3d"]
    )
    def test_rejects_bad_box_shapes(self, boxes: np.ndarray) -> None:
        with pytest.raises(ValueError, match="boxes must have shape"):
            compute_soft_nms(boxes, np.ones(len(boxes)))

    @pytest.mark.parametrize("scores", [np.ones(2), np.ones((3, 1))], ids=["short", "2d"])
    def test_rejects_scores_of_the_wrong_shape(self, scores: np.ndarray) -> None:
        with pytest.raises(ValueError, match="scores must be a 1D array of length 3"):
            compute_soft_nms(DISJOINT, scores)

    def test_rejects_non_finite_scores(self) -> None:
        with pytest.raises(ValueError, match="scores must be finite"):
            compute_soft_nms(DISJOINT, np.array([0.9, np.nan, 0.5]))

    def test_rejects_inverted_boxes(self) -> None:
        with pytest.raises(ValueError, match="x_min <= x_max"):
            compute_soft_nms(np.array([[10, 0, 0, 10]], dtype=float), np.array([0.9]))

    def test_does_not_modify_its_inputs(self) -> None:
        boxes = OVERLAPPING.copy()
        scores = np.array([0.9, 0.8, 0.7])
        compute_soft_nms(boxes, scores)
        np.testing.assert_array_equal(boxes, OVERLAPPING)
        np.testing.assert_array_equal(scores, [0.9, 0.8, 0.7])
