from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int


class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image) -> list[Detection]:
        results = self.model(image)

        detections = []

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box.int().tolist()
                detections.append(Detection(x1, y1, x2, y2))

        return detections
