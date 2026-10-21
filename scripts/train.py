"""Fine-tune a COCO-pretrained detector on RDD2022 with Ultralytics; the same settings for RT-DETR and YOLO.

usage: uv run scripts/train.py --model rtdetr-l.pt --data all --name rtdetr_all
       uv run scripts/train.py --model yolo11m.pt --data japan --name yolo_japan
"""
import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO


def main(args):
    if args.vram < 1:
        import torch

        # the same GPU drives the desktop: a full card freezes the screen, so leave room for it
        torch.cuda.set_per_process_memory_fraction(args.vram)
    model = RTDETR(args.model) if "rtdetr" in args.model else YOLO(args.model)
    if args.resume:
        # after a crash: args.model is the run's last.pt, and every setting comes from that run
        model.train(resume=True)
        return
    model.train(data=str(Path("data/yolo") / f"{args.data}.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, workers=8, seed=0, project=str(Path("runs").resolve()), name=args.name,
                exist_ok=True,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vram", type=float, default=0.8, help="max share of GPU memory for this process")
    main(parser.parse_args())
