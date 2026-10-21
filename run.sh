#!/usr/bin/env bash
# Shortcuts for this project (make isn't installed on the machine it was built on).
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-help}" in
  test)        uv run pytest -q && uv run ruff check src scripts tests ;;
  prepare)     uv run scripts/prepare.py ;;
  train)       uv run scripts/train.py --model yolo11m.pt --data all --name yolo_all
               uv run scripts/train.py --model rtdetr-l.pt --data all --name rtdetr_all
               uv run scripts/train.py --model yolo11m.pt --data japan --name yolo_japan ;;
  evaluate)    uv run scripts/evaluate.py yolo_all rtdetr_all yolo_japan ;;
  export)      uv run --group export scripts/export_benchmark.py rtdetr_all && mkdir -p data/models \
                 && cp runs/rtdetr_all/weights/best.engine data/models/rtdetr_all.engine \
                 && cp runs/rtdetr_all/weights/best.onnx data/models/rtdetr_all.onnx ;;
  gate)        uv run --group mlops scripts/quality_gate.py --register ;;
  api)         uv run --group export uvicorn road_damage_colombia.entrypoints.api:app --port 8000 ;;
  web)         cd web && PATH=$HOME/.local/opt/node/bin:$PATH npm run dev ;;
  monitoring)  docker compose -f monitoring/docker-compose.yml up -d
               echo "Grafana on http://localhost:3001, Prometheus on http://localhost:9090" ;;
  monitoring-down) docker compose -f monitoring/docker-compose.yml down ;;
  mlflow)      uv run --group mlops mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 ;;
  jenkins)     jenkins-bioclip ;;
  *)           echo "usage: ./run.sh test|prepare|train|evaluate|export|gate|api|web|monitoring|monitoring-down|mlflow|jenkins" ;;
esac
