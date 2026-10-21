# Detection: RT-DETR vs YOLO, country by country

RDD2022, four damage classes (D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole). Each
country split 70/15/15 with a fixed seed (`results/dataset.json`). The challenge's own test sets carry no labels.

- `rtdetr_all`, `yolo_all`: RT-DETR-L and YOLO11m, both COCO-pretrained, fine-tuned 20 epochs on Japan + India + Czech
  + United States (18,091 images), same Ultralytics pipeline, 640 px, batch 8.
- `yolo_japan`: YOLO11m on Japan only (7,354 images).
- **Norway is never trained on**: a stand-in for a country the model has never seen, like Colombia.

Test mAP50 (`by_country.csv` has mAP50-95, per-class AP and timings):

| model | Japan | India | Czech | United States | Norway (unseen) |
|---|---:|---:|---:|---:|---:|
| **RT-DETR-L, 4 countries** | **0.573** | **0.390** | **0.365** | **0.611** | **0.158** |
| YOLO11m, 4 countries | 0.502 | 0.319 | 0.281 | 0.560 | 0.108 |
| YOLO11m, Japan only | 0.491 | 0.069 | 0.171 | 0.418 | 0.080 |

**RT-DETR beats YOLO in every country**, by 5 to 8.4 points of mAP50 (and 1.6 to 3.1 of mAP50-95), with the same
data, epochs and augmentations: the paper's claim holds in this domain.

**A model from one country barely works in another.** Trained on Japan, YOLO keeps its accuracy at home (0.491, about
the same as the 4-country model's 0.502) and collapses in India (0.069 against 0.319). The data, not the architecture,
decides where a model works.

**No model is ready for a country it hasn't seen.** On Norway every model falls to 0.08–0.16: different cameras,
roads, light and snow-worn asphalt. For Colombia, that means labelled local photos before trusting any of these.

**Where the time goes (PyTorch, batch 16, RTX 4060):** RT-DETR spends 0.15 ms per image after the network, against
0.5–0.8 ms of NMS for YOLO, so the NMS-free design does remove that step. But its network is heavier (22.9 ms against
15.5 ms), so end to end YOLO11m is still faster in PyTorch. Exported to TensorRT FP16, RT-DETR runs at 10.5 ms per image
with the same mAP ([results/serving](../serving)).
