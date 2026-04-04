"""
Batch-process a video: YOLO + risk scores + RL escalation hint + optional RAG brief rows.
Outputs: output/annotated.mp4, output/events.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.risk_scorer import compute_risk, risk_bucket
from src.rl_policy import load_or_train_q, rl_action
from src.video_io import open_video_writer
from src.video_pipeline import SimpleBagTracker, VideoAnalyzer, draw_overlay


def _parse_names(s: str) -> set[str]:
    return {x.strip().lower() for x in s.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input video")
    ap.add_argument("--out-dir", default="output", help="Output directory")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu")
    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics weights; use your trained best.pt for missile/rocket classes",
    )
    ap.add_argument(
        "--missile-classes",
        default="missile,rocket,projectile,cruise_missile,ballistic_missile,rocket_motor",
        help="Comma-separated class names (must match your custom model)",
    )
    ap.add_argument(
        "--drone-classes",
        default="drone,uav,quadcopter,multicopter,unmanned_aerial_vehicle,fpv_drone,drone_uav",
        help="Comma-separated drone/UAV class names (custom model)",
    )
    ap.add_argument(
        "--weapon-classes",
        default="gun,knife,pistol,rifle",
        help="Extra weapon class names for custom YOLO (merged with COCO bat/racket)",
    )
    ap.add_argument(
        "--fight-classes",
        default="fight,fighting,punch,violence,brawl,assault",
        help="Fight / violence class names",
    )
    ap.add_argument(
        "--theft-classes",
        default="shoplifting,stealing,theft,burglary,robbery,pickpocket",
        help="Theft / shoplifting class names",
    )
    ap.add_argument(
        "--accident-classes",
        default="accident,car_accident,road_accident,crash,collision,vehicle_crash,wreck,damaged_vehicle",
        help="Road accident / crash class names (custom YOLO)",
    )
    ap.add_argument("--site-name", default="", help="Label on overlay / CSV metadata")
    ap.add_argument("--lat", type=float, default=None, help="Fixed camera latitude")
    ap.add_argument("--lng", type=float, default=None, help="Fixed camera longitude")
    args = ap.parse_args()
    missile_names = _parse_names(args.missile_classes)
    drone_names = _parse_names(args.drone_classes)
    weapon_names = _parse_names(args.weapon_classes)
    fight_names = _parse_names(args.fight_classes)
    theft_names = _parse_names(args.theft_classes)
    accident_names = _parse_names(args.accident_classes)

    inp = Path(args.input)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "events.csv"
    for name in ("annotated.mp4", "annotated.avi"):
        old = out_dir / name
        if old.is_file():
            try:
                old.unlink()
            except OSError:
                pass

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {inp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, out_video = open_video_writer(out_dir / "annotated.mp4", fps, w, h)

    analyzer = VideoAnalyzer(
        model_name=args.model,
        device=args.device,
        missile_class_names=missile_names,
        drone_class_names=drone_names,
        weapon_class_names=weapon_names,
        fight_class_names=fight_names,
        theft_class_names=theft_names,
        accident_class_names=accident_names,
    )
    loc_line = None
    if args.site_name.strip():
        la = args.lat if args.lat is not None else 0.0
        ln = args.lng if args.lng is not None else 0.0
        loc_line = f"{args.site_name.strip()} | {la:.5f},{ln:.5f}"
    elif args.lat is not None and args.lng is not None:
        loc_line = f"{args.lat:.5f},{args.lng:.5f}"
    tracker = SimpleBagTracker(fps=fps)
    q_path = ROOT / "models" / "q_table.npy"
    if not q_path.exists():
        from src.rl_policy import train_q_table

        q_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(q_path, train_q_table(episodes=3000))
    Q = load_or_train_q(q_path)

    rows = []
    prev_gray = None
    frame_i = 0
    cooldown = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sig = analyzer.analyze_frame(frame, prev_gray, gray, tracker)
        prev_gray = gray

        rs = compute_risk(
            person_count=sig.person_count,
            motion_score=sig.motion_score,
            proximity_cluster_score=sig.proximity_cluster_score,
            abandoned_seconds=sig.abandoned_max_seconds,
            acoustic_score=0.0,
            weapon_like_score=sig.weapon_like_score,
            missile_like_score=sig.missile_like_score,
            drone_like_score=sig.drone_like_score,
            wifi_rf_anomaly_score=0.0,
            fight_like_score=sig.fight_like_score,
            theft_like_score=sig.theft_like_score,
            accident_like_score=sig.accident_like_score,
            vehicle_count=len(sig.vehicle_boxes),
        )

        if cooldown > 0:
            cooldown -= 1
        act_id, act_name = rl_action(rs.total, min(cooldown, 3), Q)
        if act_id != 0:
            cooldown = 6  # debounce ~6 frames at sample rate

        rows.append(
            {
                "frame": frame_i,
                "t_sec": round(frame_i / fps, 3),
                "risk_total": rs.total,
                "risk_bucket": risk_bucket(rs.total),
                "person_count": sig.person_count,
                "motion": sig.motion_score,
                "proximity": sig.proximity_cluster_score,
                "abandoned_sec": sig.abandoned_max_seconds,
                "weapon_like": sig.weapon_like_score,
                "fight_like": sig.fight_like_score,
                "theft_like": sig.theft_like_score,
                "accident_like": sig.accident_like_score,
                "missile_like": sig.missile_like_score,
                "n_missile_boxes": len(sig.missile_boxes),
                "drone_like": sig.drone_like_score,
                "n_drone_boxes": len(sig.drone_boxes),
                "rl_action": act_name,
                "labels": ",".join(rs.labels),
            }
        )

        vis = draw_overlay(frame, sig, rs.total, rs.labels, location_line=loc_line)
        writer.write(vis)
        frame_i += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Wrote", out_video.resolve())
    print("Wrote", out_csv, "rows", len(df))
    if len(df):
        peak = float(df["risk_total"].max())
        mean_r = float(df["risk_total"].mean())
        dur = float(df["t_sec"].iloc[-1])
        print("\n--- Run summary (Indian CCTV / road & fight screening) ---")
        print(f"Duration ~{dur:.1f}s, frames={len(df)}, peak risk={peak:.0f}/100, mean={mean_r:.1f}/100")
        print(f"Peak bucket (RL): {risk_bucket(peak)}/10")
        fl = float(df["fight_like"].mean()) if "fight_like" in df.columns else 0.0
        mo = float(df["motion"].mean()) if "motion" in df.columns else 0.0
        print(f"Avg fight_like={fl:.2f}, avg motion={mo:.2f} (0-1 proxies — verify on video)")
        print("Train custom YOLO on local CCTV for road_rage/fight classes; demo not legal evidence.\n")


if __name__ == "__main__":
    main()
