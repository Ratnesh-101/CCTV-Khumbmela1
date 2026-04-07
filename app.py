"""
Streamlit control room: video demo, pandas/plotly viz, RAG briefs, RL actions,
help+location notify, simulated public broadcast, optional audio transient score.
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.acoustic_heuristic import transient_score_from_waveform
from src.notify import format_alert_location, notify_operator
from src.rag_ops import rag_operator_brief, retrieve_context
from src.risk_scorer import compute_risk, risk_bucket
from src.rl_policy import load_or_train_q, rl_action
from src.video_io import open_video_writer
from src.video_pipeline import SimpleBagTracker, VideoAnalyzer, draw_overlay
from src.wifi_drone_anomaly import (
    embedding_distance_anomaly,
    run_wifi_pipeline,
    synthetic_demo_frames,
)


def _human_summary_after_run(df: pd.DataFrame) -> str:
    """Plain-language explanation after a video run (for operators / demos)."""
    if df is None or df.empty:
        return "No frames were processed. Upload a short clip and click **Run analysis**."
    n = len(df)
    dur = float(df["t"].iloc[-1]) if "t" in df.columns and n else 0.0
    peak = float(df["total"].max())
    mean_r = float(df["total"].mean())
    b_peak = risk_bucket(peak)
    parts: list[str] = []
    parts.append(
        f"**What happened:** The clip was scanned frame-by-frame ({n} frames, about **{dur:.1f} s**). "
        f"The fused risk score peaked at **{peak:.0f}/100** (average **{mean_r:.1f}/100**). "
        f"The RL helper sees peak activity in **bucket {b_peak}** (0–9 scale).\n\n"
    )
    comp_rows: list[tuple[str, str]] = [
        ("Fight / violence-style signal", "fight_like"),
        ("Motion + crowding (arguments, scuffles, chaotic traffic)", "motion_anomaly"),
        ("Crowd density", "crowd_density"),
        ("Unattended object / bag risk", "abandoned_object"),
        ("Theft-style signal", "theft_like"),
        ("Weapon-style signal (unverified on default weights)", "weapon_like"),
        ("Road accident / traffic-incident hint (weak without crash-trained model)", "accident_like"),
    ]
    ranked: list[tuple[str, float]] = []
    for title, col in comp_rows:
        if col in df.columns:
            ranked.append((title, float(df[col].mean())))
    ranked.sort(key=lambda x: -x[1])
    parts.append("**Strongest average drivers of the score:**\n")
    for title, v in ranked[:4]:
        parts.append(f"- {title}: **{v:.2f}** (component is 0–1 before weighting)\n")
    parts.append("\n**Indian roads & CCTV:** ")
    fl = float(df["fight_like"].mean()) if "fight_like" in df.columns else 0.0
    mo = float(df["motion_anomaly"].mean()) if "motion_anomaly" in df.columns else 0.0
    al = float(df["accident_like"].mean()) if "accident_like" in df.columns else 0.0
    cd = float(df["crowd_density"].mean()) if "crowd_density" in df.columns else 0.0
    lbl_blob = ""
    if "labels" in df.columns:
        lbl_blob = " ".join(str(x) for x in df["labels"].astype(str))
    gathering_hit = any(
        tag in lbl_blob
        for tag in ("mass_gathering", "dense_crowd", "road_gathering_vehicles")
    )
    if gathering_hit or cd >= 0.32:
        parts.append(
            "\n**Mass gathering / crowding:** Flags or scores point to **many people close together** or **roadside crowding** "
            "(common on **busy roads**, **junctions**, **melas**, or **processions**). "
            "This uses **person detections + spacing + optional vehicles**, not a certified crowd count — **check the video** because small figures are often missed.\n\n"
        )
    if al >= 0.38:
        parts.append(
            "\n**Road / traffic:** The **accident / traffic-incident** channel is relatively high. "
            "That may be the **multi-vehicle heuristic** (not a real crash detector) or your custom **crash** boxes. **Review the video**; train classes like `accident`, `crash` on real incident frames to improve.\n\n"
        )
    if fl >= 0.4:
        parts.append(
            "Fight-style or close-proximity motion is relatively high — this pattern can appear in **road rage**, "
            "**street fights**, or crowded markets. **Always confirm on the actual video**; default COCO weights are weak proxies.\n\n"
        )
    elif mo >= 0.45:
        parts.append(
            "Motion and crowding are high — typical of **busy Indian roads**, **junction arguments**, or dense foot traffic. "
            "If you need reliable “fight” or “road rage” flags, train YOLO on your own CCTV with classes like `road_rage`, `fight`, `scuffle`.\n\n"
        )
    else:
        parts.append(
            "Signals are mostly moderate. For **road rage / fight** detection on Indian footage, upload a **custom `best.pt`** "
            "trained on local clips and map class names in the expander below.\n\n"
        )
    rl_counts = df["rl_action"].value_counts().to_dict() if "rl_action" in df.columns else {}
    if rl_counts:
        top_a = max(rl_counts, key=lambda k: rl_counts[k])
        parts.append(
            f"**Escalation hints (RL):** Most common suggested action in this clip: **{top_a}** "
            f"({rl_counts[top_a]} frames). This is a demo policy, not a deployment recommendation.\n\n"
        )
    parts.append(
        "\n**Disclaimer:** screening / demo only — not legal evidence. "
        "Do not use for automated enforcement without review and compliance checks."
    )
    return "".join(parts)


def _readable_events_tail(df: pd.DataFrame) -> pd.DataFrame:
    tail = df.tail(20).copy()
    mapping = {
        "total": "Overall risk (0–100)",
        "t": "Time (seconds)",
        "frame": "Frame number",
        "rl_action": "Suggested action (demo)",
        "crowd_density": "Crowding",
        "motion_anomaly": "Movement & crowding",
        "abandoned_object": "Left-behind bag risk",
        "weapon_like": "Possible weapon (unverified)",
        "missile_like": "Missile-like (custom model)",
        "drone_like": "Drone-like (custom model)",
        "wifi_rf_anomaly": "Wi‑Fi anomaly blend",
        "fight_like": "Fight / violence hint",
        "theft_like": "Theft hint",
        "accident_like": "Accident / traffic hint",
        "labels": "Risk flags (text)",
    }
    return tail.rename(columns={k: v for k, v in mapping.items() if k in tail.columns})


def _latest_annotated_output() -> Path | None:
    out = ROOT / "output"
    for name in ("annotated_ui.mp4", "annotated_ui.avi"):
        p = out / name
        if p.is_file() and p.stat().st_size > 32:
            return p
    return None


def _remove_prior_annotated_ui() -> None:
    out = ROOT / "output"
    for name in ("annotated_ui.mp4", "annotated_ui.avi"):
        p = out / name
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


def _ensure_q_table() -> np.ndarray:
    p = ROOT / "models" / "q_table.npy"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        from src.rl_policy import train_q_table

        np.save(p, train_q_table(episodes=2500))
    return load_or_train_q(p)


def _approx_geo():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=8)
        if r.ok:
            j = r.json()
            return {
                "lat": j.get("latitude"),
                "lng": j.get("longitude"),
                "city": j.get("city"),
                "accuracy_m": 5000,
                "source": "ip_geolocation",
            }
    except Exception:
        pass
    return None


st.set_page_config(page_title="CCTV safety screening", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 1180px; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("CCTV safety screening")
st.markdown(
    "Upload a short **CCTV clip** to see **highlighted people**, **risk flags**, and simple **charts**. "
    "Built for roads, shops, and public spaces — **not** for naming or tracking individuals."
)
with st.expander("Quick guide — start here", expanded=False):
    st.markdown(
        """
| Step | What to do |
|------|------------|
| **1** | *(Optional)* Open **Camera location** in the left **Camera & model** column and set site name and map coordinates for alerts. |
| **2** | Choose your video file, then click **Run analysis**. |
| **3** | Watch the **annotated video** and read **Your summary** below the charts. |

**Tips:** Short clips (about **10–60 seconds**) run faster on a normal PC. For **road rage** or **fights**, you will need your own trained detection model (see **Advanced model settings**).

**Other tabs:** **Get help** sends a test alert; **Playbook** searches procedure text; **Wi‑Fi** and **Audio** are optional extras.
        """
    )
with st.expander("How to run from a terminal (for installers)", expanded=False):
    st.code("streamlit run app.py", language="bash")
with st.expander("Train your own YOLO model (road rage, fights, weapons)", expanded=False):
    st.markdown(
        "You need **labelled images** (YOLO `.txt` boxes). Copy `training/dataset.example.yaml` to your dataset folder, "
        "set `path` / `names`, then run from the project root (venv active):"
    )
    st.code(
        "python scripts/train_yolo_cctv.py --data datasets/my_cctv/dataset.yaml --epochs 100 --batch 16",
        language="bash",
    )
    st.caption("Weights appear under `runs/detect/cctv_train/weights/best.pt`. Point **Model file path** there.")

if "broadcast_msg" not in st.session_state:
    st.session_state.broadcast_msg = None
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "wifi_rf_anomaly_score" not in st.session_state:
    st.session_state.wifi_rf_anomaly_score = 0.0
if "site_name" not in st.session_state:
    st.session_state.site_name = ""
if "site_lat" not in st.session_state:
    st.session_state.site_lat = 0.0
if "site_lng" not in st.session_state:
    st.session_state.site_lng = 0.0

Q = _ensure_q_table()

tab_dash, tab_help, tab_rag, tab_wifi, tab_audio = st.tabs(
    [
        "Analyze video",
        "Get help",
        "Playbook search",
        "Wi‑Fi check (optional)",
        "Audio check (optional)",
    ]
)

with tab_dash:
    st.divider()
    col_side, col_main = st.columns([1, 2.15], gap="large")

    with col_side:
        st.markdown("##### Camera & model")
        with st.expander("Camera location (for alerts)", expanded=False):
            st.markdown(
                "Videos do not include GPS. Add your **fixed camera** name and coordinates so alerts show **where** the camera is."
            )
            st.session_state.site_name = st.text_input(
                "Camera or site name",
                value=st.session_state.site_name or "Camera-01",
            )
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.site_lat = st.number_input(
                    "Latitude",
                    format="%.6f",
                    value=float(st.session_state.site_lat),
                    min_value=-90.0,
                    max_value=90.0,
                )
            with c2:
                st.session_state.site_lng = st.number_input(
                    "Longitude",
                    format="%.6f",
                    value=float(st.session_state.site_lng),
                    min_value=-180.0,
                    max_value=180.0,
                )
            if st.button("Guess location from this network (approximate city)"):
                g = _approx_geo()
                if g and g.get("lat") is not None:
                    st.session_state.site_lat = float(g["lat"])
                    st.session_state.site_lng = float(g["lng"])
                    st.success(f"Set to roughly **{g.get('city')}** (from network, not the camera).")
                    st.rerun()
                else:
                    st.warning("Could not guess location.")

        with st.expander("Aerial targets (expert / custom model)", expanded=False):
            st.info(
                "The default model does **not** spot missiles or drones. Use only if you have trained **custom** weights."
            )
            weights_path = st.text_input(
                "Model file path",
                value="models/demo_trained_yolov8n.pt",
                help="Pre-trained demo: models/demo_trained_yolov8n.pt (COCO8 fine-tune). Or yolov8n.pt / your best.pt",
            )
            missile_cls_str = st.text_input(
                "Missile-related class names (comma-separated)",
                value="missile,rocket,projectile,cruise_missile,ballistic_missile,rocket_motor",
            )
            missile_names = {x.strip().lower() for x in missile_cls_str.split(",") if x.strip()}
            drone_cls_str = st.text_input(
                "Drone class names (comma-separated)",
                value="drone,uav,quadcopter,multicopter,unmanned_aerial_vehicle,fpv_drone,drone_uav",
            )
            drone_names = {x.strip().lower() for x in drone_cls_str.split(",") if x.strip()}

        with st.expander("Weapons, fights, theft, road accidents (advanced)", expanded=False):
            st.warning(
                "**COCO does not know “accident”.** The app uses **two cues**: (1) your custom classes like `crash` / `accident` if the model has them, "
                "and (2) a **weak hint** when **several vehicles** are close plus motion — many false positives. Train YOLO on crash footage for real use."
            )
            weapon_extra = st.text_input(
                "Extra weapon class names (comma-separated)",
                value="gun,knife,pistol,rifle",
            )
            weapon_names = {x.strip().lower() for x in weapon_extra.split(",") if x.strip()}

            st.markdown("---")
            st.markdown("**New v2 threat categories** — add your custom trained class names below:")

            knife_cls_str = st.text_input(
                "Pocket knife / blade class names",
                value="knife,pocket_knife,blade,switchblade,penknife,small_blade,folding_knife,cutter",
            )
            knife_names = {x.strip().lower() for x in knife_cls_str.split(",") if x.strip()}

            gun_point_cls_str = st.text_input(
                "Gun-pointing / aimed weapon class names",
                value="gun_pointing,aimed_gun,weapon_aimed,gun_aimed,pistol_aimed,shooting_posture,gun_threat,armed_threat",
            )
            gun_pointing_names = {x.strip().lower() for x in gun_point_cls_str.split(",") if x.strip()}

            thela_cls_str = st.text_input(
                "Unauthorised thela / street vendor class names",
                value="thela,street_vendor,unauthorised_vendor,cart_vendor,hawker,unauthorized_stall,roadside_cart,street_cart",
            )
            thela_names = {x.strip().lower() for x in thela_cls_str.split(",") if x.strip()}

            bad_road_cls_str = st.text_input(
                "Bad road / road hazard class names",
                value="pothole,road_damage,road_debris,bad_road,road_hazard,broken_road,cracked_road,flooded_road,road_blockage",
            )
            bad_road_names = {x.strip().lower() for x in bad_road_cls_str.split(",") if x.strip()}
            fight_cls_str = st.text_input(
                "Fight / violence class names (comma-separated)",
                value="fight,fighting,punch,violence,brawl,assault,road_rage,scuffle,street_fight",
            )
            fight_names = {x.strip().lower() for x in fight_cls_str.split(",") if x.strip()}
            theft_cls_str = st.text_input(
                "Theft / shoplifting class names (comma-separated)",
                value="shoplifting,stealing,theft,burglary,robbery,pickpocket",
            )
            theft_names = {x.strip().lower() for x in theft_cls_str.split(",") if x.strip()}
            accident_cls_str = st.text_input(
                "Road accident / crash class names (comma-separated; must match your trained model)",
                value="accident,car_accident,road_accident,crash,collision,vehicle_crash,wreck,damaged_vehicle",
            )
            accident_names = {x.strip().lower() for x in accident_cls_str.split(",") if x.strip()}

        with st.expander("Technical stack (for developers)", expanded=False):
            st.caption(
                "YOLOv8 • optical flow & rules • small RL policy • optional LangChain RAG • optional Wi‑Fi Isolation Forest."
            )

    _overlay_help = (
        "**Green boxes** = people (`[person]`). When a **risk flag** is active on that frame, the green box is **brighter and thicker**. "
        "The green strip can show **MASS GATHERING**, **DENSE CROWD**, or **ROAD + PEOPLE GATHERING** when many people (and optionally vehicles) appear — wide CCTV often **under-counts** distant people, so confirm on the raw clip. "
        "**Orange-red** = accident/crash **class** (custom model) or **traffic_incident** / **road_accident_hint** in the green strip. **Gray** = COCO vehicles when an accident-hint is raised. "
        "Other colours: orange = bags, cyan = fight-class, red = weapon-like, yellow = theft-like, purple = drone-like. "
        "If the player stays blank, use **Download** and open the file in VLC or your default video app."
    )
    _chart_legends = {
        "crowd_density": "Crowding",
        "motion_anomaly": "Movement & crowding",
        "abandoned_object": "Left-behind bag",
        "weapon_like": "Weapon-like (unverified)",
        "missile_like": "Missile-like",
        "drone_like": "Drone-like",
        "wifi_rf_anomaly": "Wi‑Fi anomaly",
        "fight_like": "Fight / violence hint",
        "theft_like": "Theft hint",
        "accident_like": "Road accident / traffic hint",
    }

    with col_main:
        st.markdown("##### Analyze your clip")
        w1, w2, w3 = st.columns(3)
        with w1:
            st.metric(
                "Wi‑Fi signal mixed into risk",
                f"{st.session_state.wifi_rf_anomaly_score:.2f}",
                help="0 = off. Change this under the Wi‑Fi tab if you use that feature.",
            )
        with w2:
            st.caption(" ")
        with w3:
            st.caption(" ")

        up = st.file_uploader(
            "1. Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="MP4, AVI, MOV, or MKV. Shorter clips run faster.",
        )
        r1, r2 = st.columns(2)
        with r1:
            device = st.selectbox(
                "Computer / GPU",
                [None, "cpu", "0"],
                format_func=lambda x: "Automatic" if x is None else ("CPU only" if x == "cpu" else "GPU 0"),
                help="Pick **CPU only** if you have no GPU or run out of memory.",
            )
        with r2:
            max_frames = int(
                st.number_input(
                    "How many frames to scan",
                    min_value=30,
                    value=300,
                    step=30,
                    help="Lower = faster. One second is often 25–30 frames.",
                )
            )
        proc = st.button("2. Run analysis", type="primary", use_container_width=True)

        _prev_out = _latest_annotated_output()
        if _prev_out is not None and not (proc and up is not None):
            st.markdown("---")
            st.markdown("##### Last saved video")
            st.video(str(_prev_out))
            st.caption(f"File: `{_prev_out.name}`")
            try:
                st.download_button(
                    label="Download last highlighted video",
                    data=_prev_out.read_bytes(),
                    file_name=_prev_out.name,
                    mime="video/mp4" if _prev_out.suffix.lower() == ".mp4" else "video/x-msvideo",
                    key="dl_prev_annotated",
                )
            except OSError:
                pass
            st.info(_overlay_help)

        if proc and up is not None:
            tmp = ROOT / "output" / "_upload.mp4"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(up.read())
            cap = cv2.VideoCapture(str(tmp))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _remove_prior_annotated_ui()
            try:
                writer, out_actual = open_video_writer(
                    ROOT / "output" / "annotated_ui.mp4", fps, w, h
                )
            except OSError as e:
                st.error(str(e))
                cap.release()
            else:
                # Provide defaults for new class names in case the expander was not opened
                _knife_names = locals().get("knife_names") or {"knife","pocket_knife","blade","switchblade","penknife","small_blade","folding_knife","cutter"}
                _gun_pointing_names = locals().get("gun_pointing_names") or {"gun_pointing","aimed_gun","weapon_aimed","gun_aimed","pistol_aimed","shooting_posture","gun_threat","armed_threat"}
                _thela_names = locals().get("thela_names") or {"thela","street_vendor","unauthorised_vendor","cart_vendor","hawker","unauthorized_stall","roadside_cart","street_cart"}
                _bad_road_names = locals().get("bad_road_names") or {"pothole","road_damage","road_debris","bad_road","road_hazard","broken_road","cracked_road","flooded_road","road_blockage"}

                analyzer = VideoAnalyzer(
                    model_name=weights_path.strip() or "yolov8n.pt",
                    device=device if device != "0" else "0",
                    missile_class_names=missile_names,
                    drone_class_names=drone_names,
                    weapon_class_names=weapon_names,
                    fight_class_names=fight_names,
                    theft_class_names=theft_names,
                    accident_class_names=accident_names,
                    knife_class_names=_knife_names,
                    gun_pointing_class_names=_gun_pointing_names,
                    thela_class_names=_thela_names,
                    bad_road_class_names=_bad_road_names,
                )
                tracker = SimpleBagTracker(fps=fps)
                prev_gray = None
                rows = []
                cooldown = 0
                f = 0
                bar = st.progress(0)
                while f < max_frames:
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
                        wifi_rf_anomaly_score=st.session_state.wifi_rf_anomaly_score,
                        fight_like_score=sig.fight_like_score,
                        theft_like_score=sig.theft_like_score,
                        accident_like_score=sig.accident_like_score,
                        vehicle_count=len(sig.vehicle_boxes),
                        # NEW v2
                        pocket_knife_score=sig.pocket_knife_score,
                        gun_pointing_score=sig.gun_pointing_score,
                        unauthorised_thela_score=sig.unauthorised_thela_score,
                        bad_road_score=sig.bad_road_score,
                    )
                    if cooldown > 0:
                        cooldown -= 1
                    act_id, act_name = rl_action(rs.total, min(cooldown, 3), Q)
                    if act_id == 2 and rs.total >= 75:
                        st.session_state.broadcast_msg = (
                            f"UNVERIFIED elevated risk ({rs.total:.0f}). Follow staff. Labels: {','.join(rs.labels)}"
                        )
                    if act_id != 0:
                        cooldown = 5
                        lat = st.session_state.site_lat
                        lng = st.session_state.site_lng
                        loc_kw = {}
                        if abs(lat) > 1e-8 or abs(lng) > 1e-8:
                            loc_kw = {"lat": lat, "lng": lng}
                        msg = format_alert_location(
                            f"[{act_name}] risk={rs.total:.1f} labels={rs.labels} frame={f}",
                            site_name=st.session_state.site_name,
                            **loc_kw,
                        )
                        notify_operator(msg)
                    rows.append({**rs.to_dict(), "frame": f, "t": f / fps, "rl_action": act_name})
                    loc_line = None
                    sn = (st.session_state.site_name or "").strip()
                    la, ln = st.session_state.site_lat, st.session_state.site_lng
                    if sn or abs(la) > 1e-8 or abs(ln) > 1e-8:
                        loc_line = (
                            f"{sn} | {la:.5f},{ln:.5f}" if sn else f"{la:.5f},{ln:.5f}"
                        )
                    writer.write(
                        draw_overlay(frame, sig, rs.total, rs.labels, location_line=loc_line)
                    )
                    f += 1
                    bar.progress(min(1.0, f / max_frames))
                cap.release()
                writer.release()
                if not rows:
                    st.warning("No frames were read. Try another file or check that the upload is a valid video.")
                else:
                    df = pd.DataFrame(rows)
                    st.session_state.event_log = rows
                    st.success("Analysis finished. Your highlighted video and charts are below.")
                    st.markdown("---")
                    st.markdown("##### Video with boxes and risk labels")
                    if not out_actual.is_file() or out_actual.stat().st_size < 64:
                        st.error(
                            "The output file is missing or too small to play. Try fewer frames or a different clip."
                        )
                    else:
                        st.video(str(out_actual))
                        st.caption(
                            f"Saved as `{out_actual.name}` (browser-friendly codec when your PC supports it)."
                        )
                        try:
                            st.download_button(
                                label="Download highlighted video",
                                data=out_actual.read_bytes(),
                                file_name=out_actual.name,
                                mime=(
                                    "video/mp4"
                                    if out_actual.suffix.lower() == ".mp4"
                                    else "video/x-msvideo"
                                ),
                                key="dl_new_annotated",
                            )
                        except OSError:
                            pass
                    st.info(_overlay_help)

                    peak = float(df["total"].max())
                    mean_r = float(df["total"].mean())
                    dur = float(df["t"].iloc[-1]) if len(df) else 0.0
                    # --- Alert level banner ---
                    if "alert_level" in df.columns:
                        top_level = df["alert_level"].value_counts().idxmax() if not df.empty else "GREEN"
                        if top_level == "RED" or peak >= 80:
                            st.error("🔴 **RED ALERT** — One or more HIGH-RISK events detected (mass gathering / weapon / gun / thela / bad road). Verify immediately.")
                        elif top_level == "AMBER" or peak >= 50:
                            st.warning("🟡 **AMBER** — Elevated risk detected. Review the clip and take precautionary action.")
                        else:
                            st.success("🟢 **GREEN** — No significant risk flags in this clip.")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric(
                            "Highest risk in clip",
                            f"{peak:.0f} / 100",
                            help="100 = strongest combined signal in this demo.",
                        )
                    with m2:
                        st.metric("Average risk", f"{mean_r:.1f} / 100")
                    with m3:
                        st.metric("Clip length used", f"{dur:.1f} sec")

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df["t"],
                            y=df["total"],
                            mode="lines",
                            name="Overall risk",
                            line=dict(color="#1f77b4", width=2),
                        )
                    )
                    fig.update_layout(
                        title="How overall risk changes over time",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Risk score (0–100)",
                        legend_title="",
                        margin=dict(t=50, b=48),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    comp_cols = [
                        c
                        for c in (
                            "crowd_density",
                            "motion_anomaly",
                            "abandoned_object",
                            "weapon_like",
                            "missile_like",
                            "drone_like",
                            "wifi_rf_anomaly",
                            "fight_like",
                            "theft_like",
                            "accident_like",
                            "gathering_risk",
                            "pocket_knife_like",
                            "gun_pointing_like",
                            "unauthorised_thela",
                            "bad_road",
                        )
                        if c in df.columns
                    ]
                    cfig = px.area(
                        df,
                        x="t",
                        y=comp_cols,
                        title="What contributed to the score (each layer is one factor)",
                    )
                    cfig.update_layout(
                        xaxis_title="Time (seconds)",
                        yaxis_title="Strength (0–1 before mixing)",
                        legend_title="Factor",
                    )
                    for tr in cfig.data:
                        tr.name = _chart_legends.get(tr.name, tr.name.replace("_", " "))
                    st.plotly_chart(cfig, use_container_width=True)

                    st.markdown("##### Recent frames (numbers)")
                    st.caption("Last rows of the run — column titles are plain language.")
                    st.dataframe(_readable_events_tail(df), use_container_width=True, hide_index=True)

                    st.markdown("##### Your summary (plain English)")
                    st.markdown(_human_summary_after_run(df))
                    print("\n--- CCTV run summary (same as dashboard) ---")
                    print(_human_summary_after_run(df).replace("**", ""))
                    print("--- end summary ---\n")

    if st.session_state.broadcast_msg:
        st.error(st.session_state.broadcast_msg)
        if st.button("Clear public banner"):
            st.session_state.broadcast_msg = None
            st.rerun()

with tab_help:
    st.markdown("##### Send a test help alert")
    st.markdown(
        "Simulates an operator message. Location is a **rough guess** from the network (not the camera). "
        "Use the **Camera location** section on **Analyze video** for real fixed-camera coordinates."
    )
    note = st.text_input("Short message (optional)", "")
    if st.button("Send help alert"):
        geo = _approx_geo()
        if geo and geo.get("lat") is not None:
            msg = (
                f"[HELP] user_help lat={geo['lat']} lng={geo['lng']} "
                f"city={geo.get('city')} acc~{geo['accuracy_m']}m ({geo['source']}) note={note}"
            )
        else:
            msg = f"[HELP] user_help (geo unavailable) note={note}"
        st.write(notify_operator(msg))
        st.info(msg)

with tab_wifi:
    st.markdown("##### Optional: unusual Wi‑Fi traffic (drone-style demo)")
    st.markdown(
        "Looks for **unusual patterns** in Wi‑Fi-style data. High values are **not** proof of a drone — "
        "this is a lab-style add-on. Method background: "
        "[ISOT drone-anomaly project](https://github.com/isot-lab/Drone-Anomaly-Detection-Dataset-and-Unsupervised-Machine-Learning)."
    )
    mode = st.radio("Choose data", ["Built-in demo data", "Upload my CSV"], horizontal=True)
    use_embed = st.checkbox("Also use text-style embeddings (slower on big files)", value=False)
    up = None
    if mode == "Upload my CSV":
        up = st.file_uploader("CSV file (numeric feature columns)", type=["csv"])

    if st.button("Run Wi‑Fi check"):
        df_run: pd.DataFrame | None = None
        if mode == "Built-in demo data":
            df_run = synthetic_demo_frames()
        elif up is not None:
            df_run = pd.read_csv(up)
        else:
            st.warning("Choose the built-in demo or upload a CSV first.")

        if df_run is not None:
            with st.spinner("Analysing…"):
                _, result = run_wifi_pipeline(df_run)
                a = result.anomaly_01.astype(float)
                embed_ok = True
                if use_embed:
                    y = (
                        df_run["Label"].to_numpy()
                        if "Label" in df_run.columns
                        else np.zeros(len(df_run))
                    )
                    normal_m = np.isin(y, (-1, 0))
                    emb_a = embedding_distance_anomaly(df_run, normal_m)
                    if emb_a is not None:
                        a = np.clip(0.65 * a + 0.35 * emb_a[: len(a)], 0.0, 1.0)
                    else:
                        embed_ok = False
                st.session_state._wifi_last = (df_run, result, a)
                st.session_state._wifi_embed_ok = embed_ok
            st.success("Wi‑Fi check finished.")
            if use_embed and not st.session_state.get("_wifi_embed_ok", True):
                st.info("Embeddings were skipped; only the standard anomaly step ran.")

    if "_wifi_last" in st.session_state:
        df_in, result, a = st.session_state._wifi_last
        st.metric("Rows analysed", len(a))
        st.metric("Strongest unusual score (0–1)", f"{float(np.max(a)):.3f}")
        st.metric("Typical score", f"{float(np.mean(a)):.3f}")
        dfp = df_in.copy()
        dfp["wifi_anomaly_01"] = a
        figw = px.scatter(
            x=result.embedding_2d[:, 0],
            y=result.embedding_2d[:, 1],
            color=dfp["wifi_anomaly_01"],
            color_continuous_scale="RdYlBu_r",
            title="2D view of traffic rows (colour = how unusual)",
        )
        st.plotly_chart(figw, use_container_width=True)
        st.dataframe(dfp.head(30), use_container_width=True)
        apply_v = float(np.max(a))
        if st.button("Add this Wi‑Fi score to video risk", key="wifi_apply"):
            st.session_state.wifi_rf_anomaly_score = apply_v
            st.success(
                f"**Analyze video** will blend in **{apply_v:.3f}** from Wi‑Fi on the next run."
            )
        if st.button("Remove Wi‑Fi from video risk", key="wifi_clear"):
            st.session_state.wifi_rf_anomaly_score = 0.0
            st.rerun()

with tab_rag:
    st.markdown("##### Search the procedure notes (RAG)")
    st.caption("Type a question or topic. Results come from the local playbook index, not live CCTV.")
    q = st.text_input("What are you looking for?", "unattended bag procedure")
    if st.button("Search playbook"):
        st.text_area("Matching notes", retrieve_context(q, k=4), height=240)
    st.markdown("---")
    st.markdown("##### Draft a short operator brief")
    al = st.text_input("Situation tags (comma-separated)", "unattended_object,disturbance_motion")
    risk = st.slider("Assumed risk level (0–100)", 0.0, 100.0, 62.0)
    if st.button("Build brief"):
        st.text_area("Suggested brief text", rag_operator_brief(al, risk), height=320)

with tab_audio:
    st.markdown("##### Optional: loud sound spike check")
    st.warning("**Demo only** — detects sudden volume changes, **not** gunshots or specific events.")
    au = st.file_uploader("Short WAV file", type=["wav"])
    if au is not None:
        try:
            import wave

            bio = io.BytesIO(au.read())
            with wave.open(bio, "rb") as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                raw = wf.readframes(n)
                width = wf.getsampwidth()
            if width == 2:
                data = np.frombuffer(raw, dtype=np.int16)
            elif width == 4:
                data = np.frombuffer(raw, dtype=np.int32)
            else:
                data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
            score = transient_score_from_waveform(np.asarray(data), int(sr))
            st.metric("Sudden-sound score (0–1)", f"{score:.3f}")
            rs = compute_risk(
                person_count=5,
                motion_score=0.1,
                proximity_cluster_score=0.1,
                abandoned_seconds=0.0,
                acoustic_score=score,
                weapon_like_score=0.0,
                missile_like_score=0.0,
                drone_like_score=0.0,
                wifi_rf_anomaly_score=st.session_state.wifi_rf_anomaly_score,
                fight_like_score=0.0,
                theft_like_score=0.0,
                accident_like_score=0.0,
                vehicle_count=0,
            )
            st.json(rs.to_dict())
        except Exception as e:
            st.error(str(e))
