"""YOLO vegetation detection module."""

import logging

import numpy as np

from vegcover.models import Box, DetectResult

logger = logging.getLogger(__name__)

# COCO classes related to vegetation
COCO_VEGETATION_CLASSES = {
    "potted plant",
    "tree",
}

# Mapping from COCO names to friendlier names
COCO_TO_VEG_NAME = {
    "potted plant": "plant",
    "tree": "tree",
}


class VegetationDetector:
    """Detects vegetation in images using YOLO."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._model = None

    @property
    def model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: np.ndarray) -> DetectResult:
        """Detect vegetation in an image.

        Args:
            image: RGB image as numpy array.

        Returns:
            DetectResult with detected boxes.
        """
        results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)

        boxes = []
        classes = set()

        for result in results:
            if result.boxes is None:
                continue

            # Handle both tensor (with .cpu()) and numpy array mocks
            def _to_numpy(val):
                if hasattr(val, "cpu"):
                    return val.cpu().numpy()
                return val

            for box_data, conf, cls_id in zip(
                _to_numpy(result.boxes.xyxy),
                _to_numpy(result.boxes.conf),
                _to_numpy(result.boxes.cls).astype(int),
            ):
                class_name = result.names.get(cls_id, "unknown")

                # Map to vegetation-friendly name
                display_name = COCO_TO_VEG_NAME.get(class_name, class_name)

                box = Box(
                    x1=int(box_data[0]),
                    y1=int(box_data[1]),
                    x2=int(box_data[2]),
                    y2=int(box_data[3]),
                    class_name=display_name,
                    confidence=float(conf),
                )
                boxes.append(box)
                classes.add(display_name)

        return DetectResult(
            boxes=boxes,
            classes=sorted(list(classes)),
            model_name=self.model_path,
        )
