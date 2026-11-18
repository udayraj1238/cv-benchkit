"""Unit tests for cv_benchkit.converters."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from cv_benchkit.converters import (
    BBox,
    COCOAnnotation,
    COCOCategory,
    COCODataset,
    COCOImage,
    coco_to_voc,
    coco_to_yolo,
    cxcywh_to_xyxy,
    normalized_to_xyxy,
    validate_coco_file,
    voc_to_coco,
    voc_to_yolo,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_normalized,
    xyxy_to_xywh,
    yolo_to_coco,
    yolo_to_voc,
)


@pytest.mark.parametrize("converter", [coco_to_yolo, coco_to_voc])
@pytest.mark.parametrize("as_string", [False, True])
def test_coco_output_rejects_existing_file_without_changing_it(
    tmp_path: Path,
    converter: Callable[[COCODataset, Path | str], list[Path]],
    as_string: bool,
) -> None:
    output = tmp_path / "existing labels.txt"
    output.write_text("preserve this file", encoding="utf-8")
    dataset = COCODataset(images=[], annotations=[], categories=[])
    with pytest.raises(NotADirectoryError, match="output_dir cannot be an existing file") as exc:
        converter(dataset, str(output) if as_string else output)
    assert str(output) in str(exc.value)
    assert output.read_text(encoding="utf-8") == "preserve this file"


class TestBBoxConversions:
    """Tests for coordinate conversions and round-trip consistency."""

    @pytest.mark.parametrize(
        ("xyxy", "expected_xywh"),
        [
            ([10.0, 20.0, 50.0, 80.0], [10.0, 20.0, 40.0, 60.0]),
            ([0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]),
            ([5.5, 10.5, 15.5, 20.5], [5.5, 10.5, 10.0, 10.0]),
        ],
    )
    def test_xyxy_to_xywh_and_back(self, xyxy: list[float], expected_xywh: list[float]) -> None:
        arr_xyxy = np.array([xyxy])
        xywh = xyxy_to_xywh(arr_xyxy)
        np.testing.assert_allclose(xywh[0], expected_xywh)

        recovered = xywh_to_xyxy(xywh)
        np.testing.assert_allclose(recovered[0], xyxy)

    @pytest.mark.parametrize(
        ("xyxy", "expected_cxcywh"),
        [
            ([10.0, 20.0, 50.0, 80.0], [30.0, 50.0, 40.0, 60.0]),
            ([0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 100.0, 100.0]),
        ],
    )
    def test_xyxy_to_cxcywh_and_back(self, xyxy: list[float], expected_cxcywh: list[float]) -> None:
        arr_xyxy = np.array([xyxy])
        cxcywh = xyxy_to_cxcywh(arr_xyxy)
        np.testing.assert_allclose(cxcywh[0], expected_cxcywh)

        recovered = cxcywh_to_xyxy(cxcywh)
        np.testing.assert_allclose(recovered[0], xyxy)

    def test_xyxy_to_normalized_and_back(self) -> None:
        img_w, img_h = 640, 480
        boxes = np.array([[64.0, 48.0, 320.0, 240.0], [0.0, 0.0, 640.0, 480.0]])
        norm = xyxy_to_normalized(boxes, img_w=img_w, img_h=img_h)
        assert np.all(norm >= 0.0) and np.all(norm <= 1.0)

        recovered = normalized_to_xyxy(norm, img_w=img_w, img_h=img_h)
        np.testing.assert_allclose(recovered, boxes)

    def test_empty_boxes(self) -> None:
        empty = np.empty((0, 4), dtype=np.float64)
        assert xyxy_to_xywh(empty).shape == (0, 4)
        assert xywh_to_xyxy(empty).shape == (0, 4)
        assert xyxy_to_cxcywh(empty).shape == (0, 4)
        assert cxcywh_to_xyxy(empty).shape == (0, 4)
        assert xyxy_to_normalized(empty, 100, 100).shape == (0, 4)
        assert normalized_to_xyxy(empty, 100, 100).shape == (0, 4)

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError):
            xyxy_to_normalized(np.array([[10, 20, 30, 40]]), img_w=0, img_h=100)
        with pytest.raises(ValueError):
            xyxy_to_normalized(np.array([[10, 20, 30, 40]]), img_w=100, img_h=-5)

    def test_xyxy_to_xywh_inverted_boxes(self) -> None:
        inverted = np.array([[30.0, 40.0, 10.0, 20.0]])
        with pytest.raises(ValueError, match="inverted bounding box"):
            xyxy_to_xywh(inverted)

        sanitized = xyxy_to_xywh(inverted, sanitize=True)
        np.testing.assert_allclose(sanitized, np.array([[10.0, 20.0, 20.0, 20.0]]))

    @pytest.mark.parametrize(
        "invalid_box",
        [
            [10.0, 20.0, -5.0, 30.0],
            [10.0, 20.0, 30.0, -1.0],
            [10.0, 20.0, -10.0, -20.0],
            [[10.0, 20.0, 30.0, 40.0], [5.0, 5.0, -1.0, 10.0]],
        ],
    )
    def test_xywh_to_xyxy_negative_dimensions_raise(self, invalid_box: list) -> None:
        boxes = np.array(invalid_box)
        with pytest.raises(ValueError, match="Box width and height must be non-negative"):
            xywh_to_xyxy(boxes)

    @pytest.mark.parametrize(
        ("xywh", "expected_xyxy"),
        [
            ([10.0, 20.0, 0.0, 30.0], [10.0, 20.0, 10.0, 50.0]),
            ([10.0, 20.0, 30.0, 0.0], [10.0, 20.0, 40.0, 20.0]),
            ([10.0, 20.0, 0.0, 0.0], [10.0, 20.0, 10.0, 20.0]),
        ],
    )
    def test_xywh_to_xyxy_zero_dimensions_pass(
        self, xywh: list[float], expected_xyxy: list[float]
    ) -> None:
        out = xywh_to_xyxy(np.array([xywh]))
        np.testing.assert_allclose(out[0], expected_xyxy)

    @pytest.mark.parametrize(
        "invalid_box",
        [
            [50.0, 50.0, -5.0, 30.0],
            [50.0, 50.0, 30.0, -1.0],
            [50.0, 50.0, -10.0, -20.0],
            [[50.0, 50.0, 30.0, 40.0], [25.0, 25.0, -1.0, 10.0]],
        ],
    )
    def test_cxcywh_to_xyxy_negative_dimensions_raise(self, invalid_box: list) -> None:
        boxes = np.array(invalid_box)
        with pytest.raises(ValueError, match="Box width and height must be non-negative"):
            cxcywh_to_xyxy(boxes)

    @pytest.mark.parametrize(
        ("cxcywh", "expected_xyxy"),
        [
            ([50.0, 50.0, 0.0, 30.0], [50.0, 35.0, 50.0, 65.0]),
            ([50.0, 50.0, 30.0, 0.0], [35.0, 50.0, 65.0, 50.0]),
            ([50.0, 50.0, 0.0, 0.0], [50.0, 50.0, 50.0, 50.0]),
        ],
    )
    def test_cxcywh_to_xyxy_zero_dimensions_pass(
        self, cxcywh: list[float], expected_xyxy: list[float]
    ) -> None:
        out = cxcywh_to_xyxy(np.array([cxcywh]))
        np.testing.assert_allclose(out[0], expected_xyxy)


class TestBBoxPydantic:
    """Tests for Pydantic models."""

    def test_valid_bbox(self) -> None:
        b = BBox(coords=[10.0, 20.0, 30.0, 40.0], fmt="xyxy")
        assert b.coords == [10.0, 20.0, 30.0, 40.0]
        assert b.fmt == "xyxy"

    def test_invalid_length(self) -> None:
        with pytest.raises(ValidationError):
            BBox(coords=[10.0, 20.0, 30.0], fmt="xyxy")

    def test_invalid_format(self) -> None:
        with pytest.raises(ValidationError):
            BBox(coords=[10.0, 20.0, 30.0, 40.0], fmt="invalid_format")


class TestDatasetConversions:
    """Tests for COCO, YOLO, and VOC dataset translation."""

    @pytest.fixture
    def sample_coco_dataset(self, tmp_path: Path) -> tuple[Path, COCODataset]:
        coco_content = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 800, "height": 600},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [100.0, 100.0, 50.0, 80.0],
                    "area": 4000.0,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [200.0, 150.0, 100.0, 120.0],
                    "area": 12000.0,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 0, "name": "person"},
                {"id": 1, "name": "car"},
            ],
        }
        coco_path = tmp_path / "annotations.json"
        coco_path.write_text(json.dumps(coco_content), encoding="utf-8")
        ds = validate_coco_file(coco_path)
        return coco_path, ds

    def test_coco_to_yolo_and_back(
        self, sample_coco_dataset: tuple[Path, COCODataset], tmp_path: Path
    ) -> None:
        _coco_path, coco = sample_coco_dataset
        yolo_dir = tmp_path / "yolo_labels"

        written = coco_to_yolo(coco, yolo_dir)
        assert len(written) == 2
        assert (yolo_dir / "img1.txt").exists()
        assert (yolo_dir / "img2.txt").exists()

        # Check YOLO format content
        lines = (yolo_dir / "img1.txt").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parts = [float(p) for p in lines[0].split()]
        assert int(parts[0]) == 0
        assert all(0.0 <= p <= 1.0 for p in parts[1:])

        # Convert back
        image_sizes = {"img1": (640, 480), "img2": (800, 600)}
        cat_names = ["person", "car"]
        rec_coco = yolo_to_coco(yolo_dir, image_sizes=image_sizes, category_names=cat_names)
        assert len(rec_coco.annotations) == 2
        assert len(rec_coco.categories) == 2

    def test_coco_to_voc_and_back(
        self, sample_coco_dataset: tuple[Path, COCODataset], tmp_path: Path
    ) -> None:
        _coco_path, coco = sample_coco_dataset
        voc_dir = tmp_path / "voc_annotations"
        written = coco_to_voc(coco, voc_dir)
        assert len(written) == 2
        assert (voc_dir / "img1.xml").exists()
        assert (voc_dir / "img2.xml").exists()

        cat_names = ["person", "car"]
        rec_coco = voc_to_coco(voc_dir, category_names=cat_names)
        assert len(rec_coco.images) == 2
        assert len(rec_coco.annotations) == 2

    def test_yolo_to_voc_and_back(
        self, sample_coco_dataset: tuple[Path, COCODataset], tmp_path: Path
    ) -> None:
        _coco_path, coco = sample_coco_dataset
        yolo_dir = tmp_path / "yolo"
        coco_to_yolo(coco, yolo_dir)

        voc_dir = tmp_path / "voc"
        img_sizes = {"img1": (640, 480), "img2": (800, 600)}
        cat_names = ["person", "car"]
        voc_files = yolo_to_voc(yolo_dir, voc_dir, image_sizes=img_sizes, category_names=cat_names)
        assert len(voc_files) == 2
        assert (voc_dir / "img1.xml").exists()

        yolo_rec_dir = tmp_path / "yolo_rec"
        yolo_files = voc_to_yolo(voc_dir, yolo_rec_dir, category_names=cat_names)
        assert len(yolo_files) == 2
        assert (yolo_rec_dir / "img1.txt").exists()


@pytest.mark.parametrize(
    "width,height",
    [(0, 480), (-1, 480), (640, 0), (640, -1)],
)
@pytest.mark.parametrize("with_label", [False, True])
def test_yolo_to_coco_rejects_invalid_image_dimensions(
    tmp_path: Path, width: int, height: int, with_label: bool
) -> None:
    if with_label:
        (tmp_path / "invalid.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid.*positive"):
        yolo_to_coco(tmp_path, {"invalid": (width, height)}, ["object"])


class TestCOCOAnnotation:
    @pytest.mark.parametrize("area", [-0.25, float("nan"), float("inf"), float("-inf")])
    def test_rejects_invalid_area(self, area: float) -> None:
        with pytest.raises(ValidationError) as exc_info:
            COCOAnnotation(id=1, image_id=1, category_id=1, bbox=[0, 0, 5, 4], area=area)

        assert exc_info.value.errors()[0]["loc"] == ("area",)

    @pytest.mark.parametrize("area", [0.0, 12.5])
    def test_accepts_non_negative_area(self, area: float) -> None:
        annotation = COCOAnnotation(id=1, image_id=1, category_id=1, bbox=[0, 0, 5, 4], area=area)
        assert annotation.area == area


class TestCOCOImage:
    """Tests for COCOImage positive dimension validation (#44)."""

    @pytest.mark.parametrize(
        ("width", "height"),
        [(0, 100), (100, -1), (0, 0), (-5, 100), (100, 0)],
    )
    def test_rejects_non_positive_dimensions(self, width: int, height: int) -> None:
        with pytest.raises(ValidationError, match="image dimensions must be positive"):
            COCOImage(id=1, file_name="img.jpg", width=width, height=height)

    def test_accepts_positive_dimensions(self) -> None:
        img = COCOImage(id=1, file_name="img.jpg", width=100, height=200)
        assert img.width == 100
        assert img.height == 200


class TestCOCOCategory:
    """Tests for COCOCategory id and name validation."""

    def test_accepts_valid_category(self) -> None:
        cat = COCOCategory(id=1, name="person")
        assert cat.id == 1
        assert cat.name == "person"

    def test_rejects_negative_id(self) -> None:
        with pytest.raises(ValidationError):
            COCOCategory(id=-1, name="person")

    @pytest.mark.parametrize("name", ["", "   ", "	", "
"])
    def test_rejects_empty_or_whitespace_name(self, name: str) -> None:
        with pytest.raises(ValidationError, match="category name must be a non-empty"):
            COCOCategory(id=1, name=name)

    def test_accepts_name_with_whitespace_around(self) -> None:
        cat = COCOCategory(id=0, name=" person ")
        assert cat.name == " person "
