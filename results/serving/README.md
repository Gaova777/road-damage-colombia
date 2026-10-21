# Serving: ONNX and TensorRT

`scripts/export_benchmark.py`. Latency per image (batch 1, 640 px, median over 200 Japan test images after warm-up,
including pre- and post-processing) on an RTX 4060 laptop-class card (8 GB), and mAP50 on the Japan test set after export.

| model | format | mAP50 | median ms | p95 ms | FPS | file |
|---|---|---:|---:|---:|---:|---:|
| RT-DETR-L | PyTorch FP32 | 0.573 | 38.0 | 41.3 | 26 | 66 MB |
| RT-DETR-L | ONNX FP32 (ONNX Runtime, CUDA) | 0.573 | 26.0 | 29.1 | 38 | 132 MB |
| RT-DETR-L | **TensorRT FP16** | **0.573** | **10.5** | 12.2 | **95** | 68 MB |
| YOLO11m | PyTorch FP32 | 0.502 | 21.2 | 23.4 | 47 | 41 MB |
| YOLO11m | ONNX FP32 | 0.482 | 17.1 | 20.2 | 59 | 80 MB |
| YOLO11m | **TensorRT FP16** | 0.483 | **8.0** | 9.6 | **126** | 44 MB |

- **TensorRT FP16 makes RT-DETR 3.6x faster with no loss** (0.5732 → 0.5730).
- **Deployed, RT-DETR is 9 points more accurate than YOLO11m for 2.5 ms more per image.** Both are far above real
  time for a camera on a vehicle.
- **YOLO lost 2 points on export, and it wasn't FP16:** ONNX FP32 loses the same. The exported model takes a fixed
  640x640 square, while PyTorch validates YOLO on rectangular letterboxed batches. RT-DETR always works on the square,
  so it has nothing to lose.
