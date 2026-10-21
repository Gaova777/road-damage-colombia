"""HTTP API.

POST /detect     a road photo (and optionally where it was taken) -> damage, severity, repair priority; saved as a report
GET  /reports    every report, worst first, for the map
GET  /metrics    Prometheus: latency, detections per class, confidence, and input brightness as a cheap drift signal
GET  /health
"""
import io
import os
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from road_damage_colombia.domain.damage import NAMES, TREATMENT, priority, severity

WEIGHTS = Path(os.environ.get("RDC_WEIGHTS", "data/models/rtdetr_all.engine"))
REPORTS = Path(os.environ.get("RDC_REPORTS", "data/reports.sqlite"))

LATENCY = Histogram("rdc_inference_seconds", "Model time per photo", buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.5))
DETECTIONS = Counter("rdc_detections_total", "Damage found, by class", ["cls"])
CONFIDENCE = Histogram("rdc_detection_confidence", "Confidence of each detection", buckets=[i / 10 for i in range(1, 11)])
PHOTOS = Counter("rdc_photos_total", "Photos analysed, by priority level", ["level"])
BRIGHTNESS = Histogram("rdc_input_brightness", "Mean brightness of incoming photos (0-255); a shift means new conditions",
                       buckets=(40, 60, 80, 100, 120, 140, 160, 180, 200))
RESOLUTION = Gauge("rdc_input_megapixels", "Size of the last photo, in megapixels")

app = FastAPI(title="road-damage-colombia")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])


@lru_cache
def detector():
    from road_damage_colombia.adapters.ultralytics_detector import UltralyticsDetector

    return UltralyticsDetector(WEIGHTS)


@lru_cache
def reports():
    from road_damage_colombia.adapters.reports_sqlite import Reports

    return Reports(REPORTS)


@app.get("/health")
def health():
    return {"model": detector().name}


@app.post("/detect")
async def detect(photo: Annotated[UploadFile, File()], lat: Annotated[float | None, Form()] = None,
                 lng: Annotated[float | None, Form()] = None):
    try:
        image = Image.open(io.BytesIO(await photo.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"not an image: {e}") from e
    BRIGHTNESS.observe(float(np.asarray(image.convert("L")).mean()))
    RESOLUTION.set(image.width * image.height / 1e6)
    start = time.perf_counter()
    found = detector().detect(image)
    LATENCY.observe(time.perf_counter() - start)
    for d in found:
        DETECTIONS.labels(d.cls).inc()
        CONFIDENCE.observe(d.confidence)
    result = priority(found)
    PHOTOS.labels(result["level"]).inc()
    result["detections"] = [asdict(d) | {"name": NAMES[d.cls], "treatment": TREATMENT[d.cls],
                                         "severity": round(severity(d), 2), "in_lane": d.in_lane} for d in found]
    result["id"] = reports().add(lat, lng, result) if lat is not None and lng is not None else None
    return result


@app.get("/reports")
def list_reports():
    return reports().all()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
