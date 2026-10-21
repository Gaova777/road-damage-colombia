"""Model quality gate: evaluate a model on a fixed slice of the test sets, log the run to MLflow, and fail the build
if it got worse than quality_gate.json allows. With --register, a model that passes becomes a new version of
"road-damage-detector" in the MLflow registry.

usage: uv run --group mlops scripts/quality_gate.py --weights runs/rtdetr_all/weights/best.pt --register
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from ultralytics import RTDETR, YOLO


def check(metrics: dict, limits: dict) -> list[str]:
    failures = []
    for name, limit in limits.items():
        value = metrics[name]
        if "min" in limit and value < limit["min"]:
            failures.append(f"{name} = {value:.3f}, below {limit['min']}")
        if "max" in limit and value > limit["max"]:
            failures.append(f"{name} = {value:.3f}, above {limit['max']}")
    return failures


def main(args):
    rtdetr = "rtdetr" in str(args.weights)
    model = RTDETR(str(args.weights)) if rtdetr else YOLO(str(args.weights))
    metrics = {}
    for key, country in (("map50_japan", "Japan"), ("map50_unseen_norway", "Norway")):
        m = model.val(data=f"data/yolo/test_{country}.yaml", split="val", imgsz=640, batch=16, plots=False,
                      verbose=False, project=str(Path("runs/eval").resolve()), name=f"gate_{country}", exist_ok=True)
        metrics[key] = float(m.box.map50)
    images = sorted(Path("data/yolo/Japan/test/images").glob("*.jpg"))[:60]
    times = []
    for i, img in enumerate(images):
        start = time.perf_counter()
        model.predict(img, imgsz=640, verbose=False)
        if i >= 10:  # skip warm-up
            times.append((time.perf_counter() - start) * 1000)
    metrics["median_latency_ms"] = float(np.median(times))
    failures = check(metrics, json.loads(args.limits.read_text()))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("road-damage-quality-gate")
    with mlflow.start_run(run_name=args.weights.parent.parent.name):
        mlflow.log_params({"weights": str(args.weights), "architecture": "rtdetr" if rtdetr else "yolo"})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("gate", "failed" if failures else "passed")
        if args.register and not failures:
            mlflow.log_artifact(str(args.weights), artifact_path="model")
            run = mlflow.active_run().info.run_id
            mlflow.register_model(f"runs:/{run}/model", "road-damage-detector")
    for k, v in metrics.items():
        print(f"  {k:22} {v:.3f}")
    if failures:
        print("quality gate FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("quality gate passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("runs/rtdetr_all/weights/best.pt"))
    parser.add_argument("--limits", type=Path, default=Path("quality_gate.json"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--register", action="store_true")
    main(parser.parse_args())
