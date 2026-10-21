"""Export the chosen model to ONNX and TensorRT (FP16) and measure what each costs and keeps:
latency per image on this GPU and mAP50 on the combined test set.

usage: uv run scripts/export_benchmark.py rtdetr_all
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import RTDETR, YOLO


def load(path, rtdetr):
    return RTDETR(str(path)) if rtdetr else YOLO(str(path))


def latency(model, images, imgsz, warmup=10):
    for img in images[:warmup]:
        model.predict(img, imgsz=imgsz, verbose=False)
    times = []
    for img in images:
        start = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": round(float(np.median(times)), 2), "p95_ms": round(float(np.percentile(times, 95)), 2),
            "fps": round(1000 / float(np.median(times)), 1)}


def main(args):
    rtdetr = args.run.startswith("rtdetr")
    best = Path("runs") / args.run / "weights" / "best.pt"
    images = sorted(Path("data/yolo/Japan/test/images").glob("*.jpg"))[: args.n]
    variants = {"pytorch_fp32": best}
    variants["onnx_fp32"] = Path(load(best, rtdetr).export(format="onnx", imgsz=args.imgsz, dynamic=False))
    variants["tensorrt_fp16"] = Path(load(best, rtdetr).export(format="engine", imgsz=args.imgsz, half=True))
    out = {}
    for name, path in variants.items():
        model = load(path, rtdetr)
        m = model.val(data="data/yolo/test_Japan.yaml", split="val", imgsz=args.imgsz, batch=1, plots=False,
                      verbose=False, project=str(Path("runs/eval").resolve()), name=f"export_{name}", exist_ok=True)
        out[name] = {"map50_japan_test": round(float(m.box.map50), 4), "size_mb": round(path.stat().st_size / 1e6, 1),
                     **latency(model, images, args.imgsz)}
        print(name, out[name], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"export_{args.run}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=Path("results/serving"))
    main(parser.parse_args())
