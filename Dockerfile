# The API on CPU with the ONNX export (no GPU needed to try it). Build after ./run.sh export.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --no-dev --frozen && uv pip install onnxruntime
COPY data/models/rtdetr_all.onnx data/models/rtdetr_all.onnx
ENV RDC_WEIGHTS=data/models/rtdetr_all.onnx RDC_REPORTS=/data/reports.sqlite
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "road_damage_colombia.entrypoints.api:app", "--host", "0.0.0.0", "--port", "8000"]
