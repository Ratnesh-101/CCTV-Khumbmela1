#!/usr/bin/env python3
"""
Train a YOLOv8 detection model for CCTV (fights, road rage, weapons, bags, etc.).

Prerequisites:
  1. Label images in YOLO format (one .txt per image with class cx cy w h normalized).
  2. Copy training/dataset.example.yaml, set `path`, train/val folders, and `names` to match your labels.

Run (from project root, venv active):
  python scripts/train_yolo_cctv.py --data path/to/your/dataset.yaml

Output:
  Ultralytics writes weights under runs/detect/<name>/weights/best.pt
  Point the Streamlit app "Model file path" to that best.pt and align class name fields.

This does NOT train the small RL Q-table (that auto-trains on first app run) or Wi-Fi models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train YOLOv8 for CCTV risk demo (Ultralytics).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--data",
        type=Path,
        required=True,
        help="dataset.yaml (see training/dataset.example.yaml)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base weights: yolov8n.pt, yolov8s.pt, yolov8m.pt, or a .pt checkpoint",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="Reduce if GPU runs out of memory")
    ap.add_argument("--device", type=str, default="", help="e.g. 0 or cpu (empty = auto)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="cctv_train")
    ap.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without val improvement)",
    )
    args = ap.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"dataset yaml not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kw = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / args.project),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        exist_ok=True,
    )
    if args.device:
        train_kw["device"] = args.device

    print("Training with:", train_kw)
    model.train(**train_kw)

    out = ROOT / args.project / args.name / "weights" / "best.pt"
    print("\nDone.")
    print("Best weights:", out.resolve())
    print("In the app, set Model file path to this file and match class names in Advanced expanders.")


if __name__ == "__main__":
    main()
