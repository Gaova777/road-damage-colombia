"""Fine-tune a COCO-pretrained detector on RDD2022 with Ultralytics; the same settings for RT-DETR and YOLO.

usage: uv run scripts/train.py --model rtdetr-l.pt --data all --name rtdetr_all
       uv run scripts/train.py --model yolo11m.pt --data japan --name yolo_japan
"""
import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO


def main(args):
    model = RTDETR(args.model) if "rtdetr" in args.model else YOLO(args.model)
    model.train(data=str(Path("data/yolo") / f"{args.data}.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, workers=8, seed=0, project="runs", name=args.name, exist_ok=True,
                patience=args.patience, plots=True, amp=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", choices=["all", "japan"], default="all")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    main(parser.parse_args())
