"""Evaluate trained models on every country's held-out test set: mAP, per-class AP50, and time per image
(pre-processing, network, post-processing: the NMS step RT-DETR claims to remove).

usage: uv run scripts/evaluate.py rtdetr_all yolo_all yolo_japan
"""
import argparse
import csv
import json
from pathlib import Path

from ultralytics import RTDETR, YOLO

from road_damage_colombia.domain.damage import CLASSES

COUNTRIES = ["Japan", "India", "Czech", "United_States", "Norway"]


def main(args):
    rows = []
    for name in args.runs:
        weights = Path("runs") / name / "weights" / "best.pt"
        model = RTDETR(str(weights)) if name.startswith("rtdetr") else YOLO(str(weights))
        for country in COUNTRIES:
            m = model.val(data=str(Path("data/yolo") / f"test_{country}.yaml"), split="val", imgsz=args.imgsz,
                          batch=16, plots=False, verbose=False, project="runs/eval", name=f"{name}_{country}",
                          exist_ok=True)
            ap50 = dict(zip(CLASSES, (round(float(x), 4) for x in m.box.ap50)))
            row = {"model": name, "country": country, "map50": round(float(m.box.map50), 4),
                   "map50_95": round(float(m.box.map), 4), **{f"ap50_{c}": ap50.get(c) for c in CLASSES},
                   **{f"ms_{k}": round(v, 2) for k, v in m.speed.items()}}
            rows.append(row)
            print(f"  {name:12} {country:14} mAP50 {row['map50']:.3f}  mAP50-95 {row['map50_95']:.3f}  "
                  f"{sum(m.speed.values()):.1f} ms/img", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "by_country.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "by_country.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=Path("results/detection"))
    main(parser.parse_args())
