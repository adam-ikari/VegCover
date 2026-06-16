import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from vegcover.detector import VegetationDetector
from vegcover.models import DetectResult


class TestVegetationDetector:
    def test_init_default_model(self):
        detector = VegetationDetector()
        assert detector.model_path == "yolov8n.pt"
        assert detector.confidence_threshold == 0.25
        assert detector.iou_threshold == 0.45

    def test_init_custom_params(self):
        detector = VegetationDetector(
            model_path="custom.pt", confidence_threshold=0.5, iou_threshold=0.6
        )
        assert detector.model_path == "custom.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.6

    @patch("ultralytics.YOLO")
    def test_detect_returns_detect_result(self, mock_yolo_class):
        """Mock YOLO and verify detect returns proper DetectResult."""
        # Setup mock
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock results
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = np.array([[10, 20, 30, 40]])
        mock_result.boxes.conf = np.array([0.95])
        mock_result.boxes.cls = np.array([0])
        mock_result.names = {0: "tree"}
        mock_model.return_value = [mock_result]

        detector = VegetationDetector(model_path="yolov8n.pt")
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = detector.detect(img)

        assert isinstance(result, DetectResult)
        assert len(result.boxes) == 1
        assert result.boxes[0].class_name == "tree"
        assert result.boxes[0].confidence == pytest.approx(0.95)

    def test_coco_vegetation_classes(self):
        """Verify COCO vegetation-related class names are recognized."""
        from vegcover.detector import COCO_VEGETATION_CLASSES
        assert "potted plant" in COCO_VEGETATION_CLASSES
        assert "tree" in COCO_VEGETATION_CLASSES
