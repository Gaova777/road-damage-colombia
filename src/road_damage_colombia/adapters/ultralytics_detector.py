"""A trained RT-DETR or YOLO model (PyTorch, ONNX or TensorRT engine) behind the Detector port."""
from pathlib import Path

from road_damage_colombia.domain.damage import CLASSES, Detection


class UltralyticsDetector:
    def __init__(self, weights: Path, imgsz: int = 640, conf: float = 0.25):
        from ultralytics import RTDETR, YOLO

        self.name = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
        self.model = RTDETR(str(weights)) if "rtdetr" in str(weights) else YOLO(str(weights))
        self.imgsz, self.conf = imgsz, conf

    def detect(self, image) -> list[Detection]:
        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        for box, cls, conf in zip(result.boxes.xyxyn.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            out.append(Detection(CLASSES[int(cls)], round(float(conf), 3), tuple(round(v, 4) for v in box)))
        return out
