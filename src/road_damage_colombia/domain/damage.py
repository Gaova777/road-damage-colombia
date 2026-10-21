"""From detected damage to a repair decision. Plain Python: no model, no image library.

RDD2022's four classes and the usual treatment for each (crack sealing for isolated cracks, patching for alligator
cracking, pothole repair). The weights are a stated policy, not a measurement: a city would tune them.
"""
from dataclasses import dataclass

CLASSES = ["D00", "D10", "D20", "D40"]
NAMES = {"D00": "longitudinal crack", "D10": "transverse crack", "D20": "alligator crack", "D40": "pothole"}
TREATMENT = {"D00": "crack sealing", "D10": "crack sealing", "D20": "patching", "D40": "pothole repair"}
# how much worse each damage is for safety and for how fast it grows if left alone
WEIGHT = {"D00": 1.0, "D10": 1.0, "D20": 2.0, "D40": 3.0}
LEVELS = [(0.0, "low"), (1.5, "medium"), (3.0, "high"), (4.5, "urgent")]


@dataclass
class Detection:
    cls: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, as fractions of the image (0-1)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def in_lane(self) -> bool:
        """Dashcam photos: the lane ahead is the lower-middle of the image."""
        x1, y1, x2, y2 = self.box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return 0.2 <= cx <= 0.8 and cy >= 0.45


def severity(d: Detection) -> float:
    """Class weight, scaled up with size (a pothole covering 5% of the frame is worse than one at 0.2%) and in the lane."""
    size = min(1.0 + 20 * d.area, 3.0)
    lane = 1.5 if d.in_lane else 1.0
    return WEIGHT[d.cls] * size * lane * d.confidence


def priority(detections: list[Detection]) -> dict:
    """Score of a photo: its worst damage, plus a little for each additional one."""
    if not detections:
        return {"score": 0.0, "level": "none", "treatments": {}}
    scores = sorted((severity(d) for d in detections), reverse=True)
    score = round(scores[0] + 0.25 * sum(scores[1:]), 2)
    level = [name for threshold, name in LEVELS if score >= threshold][-1]
    treatments = {}
    for d in detections:
        t = treatments.setdefault(TREATMENT[d.cls], {"count": 0, "image_share": 0.0})
        t["count"] += 1
        t["image_share"] = round(t["image_share"] + d.area, 4)
    return {"score": score, "level": level, "treatments": treatments}


def cost(treatments: dict, unit_prices: dict | None) -> dict | None:
    """Only with prices a maintenance office supplies ({treatment: price per unit of image share}); no made-up defaults."""
    if not unit_prices:
        return None
    return {t: round(v["image_share"] * unit_prices[t], 2) for t, v in treatments.items() if t in unit_prices}
