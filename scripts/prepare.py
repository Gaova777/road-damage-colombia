"""Unpack RDD2022, convert its VOC boxes to YOLO labels and split each country 70/15/15.

Writes data/yolo/<country>/{train,val,test}/{images,labels} (images are symlinks) and the dataset files the trainings
use: data/yolo/all.yaml (Japan, India, Czech, United States) and data/yolo/japan.yaml.

usage: uv run scripts/prepare.py
"""
import argparse
import json
import random
import zipfile
from pathlib import Path

from road_damage_colombia.adapters.rdd2022 import read_voc, to_yolo
from road_damage_colombia.domain.damage import CLASSES

COUNTRIES = ["Japan", "India", "Czech", "United_States", "Norway"]
TRAIN_ON = {"all": ["Japan", "India", "Czech", "United_States"], "japan": ["Japan"]}


def unpack(args):
    if (args.raw / "RDD2022").exists():
        return
    with zipfile.ZipFile(args.raw / "RDD2022.zip") as z:
        z.extractall(args.raw)
    for inner in (args.raw / "RDD2022").glob("*.zip"):  # one zip per country inside the big one
        with zipfile.ZipFile(inner) as z:
            z.extractall(args.raw / "RDD2022")
        inner.unlink()


def main(args):
    unpack(args)
    rng = random.Random(args.seed)
    stats = {}
    for country in COUNTRIES:
        xmls = sorted((args.raw / "RDD2022").glob(f"{country}/train/annotations/xmls/*.xml"))
        rng.shuffle(xmls)
        n = len(xmls)
        parts = {"train": xmls[: int(0.7 * n)], "val": xmls[int(0.7 * n): int(0.85 * n)], "test": xmls[int(0.85 * n):]}
        stats[country] = {}
        for part, files in parts.items():
            img_dir = args.out / country / part / "images"
            lbl_dir = args.out / country / part / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            counts = dict.fromkeys(CLASSES, 0)
            for xml in files:
                image = xml.parents[2] / "images" / f"{xml.stem}.jpg"
                if not image.exists():
                    continue
                w, h, boxes = read_voc(xml)
                for b in boxes:
                    counts[b[0]] += 1
                link = img_dir / image.name
                if not link.exists():
                    link.symlink_to(image.resolve())
                (lbl_dir / f"{xml.stem}.txt").write_text("\n".join(to_yolo(w, h, boxes)))
            stats[country][part] = {"images": len(files), **counts}
        print(country, json.dumps(stats[country]))
    for name, countries in TRAIN_ON.items():
        lines = [f"path: {args.out.resolve()}", "train:"] + [f"  - {c}/train/images" for c in countries]
        lines += ["val:"] + [f"  - {c}/val/images" for c in countries]
        lines += ["names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
        (args.out / f"{name}.yaml").write_text("\n".join(lines) + "\n")
    for country in COUNTRIES:  # one file per test set, for evaluation
        lines = [f"path: {args.out.resolve()}", f"train: {country}/train/images", f"val: {country}/test/images",
                 "names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
        (args.out / f"test_{country}.yaml").write_text("\n".join(lines) + "\n")
    Path("results").mkdir(exist_ok=True)
    Path("results/dataset.json").write_text(json.dumps(stats, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo"))
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
