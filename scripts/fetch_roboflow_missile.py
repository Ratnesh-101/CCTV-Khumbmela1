"""
Download the Roboflow Universe dataset:
  https://universe.roboflow.com/karthii/missile-detction/dataset/1

Workspace: karthii | Project: missile-detction | Version: 1

Setup:
  1. pip install roboflow
  2. Get API key: https://app.roboflow.com/settings/api
  3. Set env:  set ROBOFLOW_API_KEY=your_key   (PowerShell: $env:ROBOFLOW_API_KEY="...")

Run (from repo root cctv-risk-demo):
  python scripts/fetch_roboflow_missile.py

Then train YOLOv8 (Ultralytics):
  yolo detect train data=datasets/missile-detction-1/data.yaml model=yolov8n.pt epochs=50 imgsz=640 project=runs name=missile_v1

Note: Roboflow names the folder like missile-detction-1 — check datasets/ after download.
Match Streamlit / CLI missile class names to the `names:` list inside data.yaml.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Karthii missile-detction v1 from Roboflow")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY)",
    )
    ap.add_argument(
        "--format",
        default="yolov8",
        help="Roboflow export format (yolov8 recommended for Ultralytics)",
    )
    args = ap.parse_args()
    if not args.api_key:
        print("Missing API key. Set ROBOFLOW_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    DATASETS.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("karthii").project("missile-detction")
    version = project.version(1)
    dataset = version.download(args.format, location=str(DATASETS))

    loc = getattr(dataset, "location", None) or str(DATASETS)
    print("Downloaded to:", loc)
    yaml_path = Path(loc) / "data.yaml"
    if not yaml_path.is_file():
        found = sorted(DATASETS.rglob("data.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        yaml_path = found[0] if found else yaml_path
    if yaml_path.is_file():
        print("data.yaml:", yaml_path)
        print("\nTrain:")
        print(
            f'  yolo detect train data="{yaml_path.as_posix()}" '
            f"model=yolov8n.pt epochs=50 imgsz=640 project=runs name=missile_rf"
        )
        print("\nUse weights in this demo (after training):")
        print('  runs/detect/missile_rf/weights/best.pt')
        print("\nOpen data.yaml and use the exact strings under `names` in the app's missile class list.")
    else:
        print("Warning: data.yaml not found at expected path; list datasets/ manually.")


if __name__ == "__main__":
    main()
