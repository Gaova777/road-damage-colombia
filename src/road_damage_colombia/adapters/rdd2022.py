"""RDD2022 (Arya et al., 2022, CC BY 4.0): Pascal VOC boxes per image, one folder per country."""
import xml.etree.ElementTree as ET
from pathlib import Path

from road_damage_colombia.domain.damage import CLASSES


def read_voc(path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """(width, height, [(class, x1, y1, x2, y2) in pixels]) keeping only the four damage classes."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    boxes = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASSES:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((name, x1, y1, x2, y2))
    return w, h, boxes


def to_yolo(w: int, h: int, boxes) -> list[str]:
    """YOLO label lines: class index and box centre/size as fractions of the image."""
    lines = []
    for name, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{CLASSES.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines
