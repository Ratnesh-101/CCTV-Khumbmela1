"""
Deep learning: Ultralytics YOLO for persons + luggage classes.
Classical CV: optical flow + frame diff for motion anomaly.
Heuristic tracking for unattended bags (centroid + idle time).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# COCO IDs via ultralytics names — we resolve at runtime from model.names
TARGET_CLASSES_DEFAULT = {"person", "backpack", "handbag", "suitcase"}


def _c01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def fight_heuristic_score(
    motion: float, proximity: float, person_count: int
) -> float:
    """Proxy only: close people + motion — not a real fight classifier."""
    if person_count < 2:
        return 0.0
    return _c01(0.55 * motion + 0.45 * proximity)


def theft_heuristic_score(
    abandoned_sec: float,
    motion: float,
    person_count: int,
    bag_count: int,
) -> float:
    """Proxy: unattended bags + activity — train custom YOLO for real use."""
    if bag_count < 1:
        return 0.0
    ab = _c01(abandoned_sec / 55.0)
    pc = _c01(person_count / 6.0)
    return _c01(ab * 0.5 + motion * 0.3 + pc * 0.2)


def vehicle_proximity_score(
    vehicle_boxes: List[Tuple[float, float, float, float]],
) -> float:
    """0–1: several road vehicles close in the frame — weak cue only, not a crash."""
    if len(vehicle_boxes) < 2:
        return 0.0
    cents = [_centroid(b) for b in vehicle_boxes]
    close = 0
    pairs = 0
    thr = 150.0
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            pairs += 1
            if _dist(cents[i], cents[j]) < thr:
                close += 1
    if pairs == 0:
        return 0.0
    return float(min(1.0, (close / pairs) * 2.8))


def accident_heuristic_score(
    vehicle_count: int,
    vehicle_prox: float,
    motion: float,
    person_count: int,
) -> float:
    """
    Demo-only proxy for *possible* road incidents: multiple COCO vehicles nearby + motion + people.
    Real accident detection needs a model trained on crashes / damaged vehicles.
    """
    if vehicle_count < 2:
        return 0.0
    vp = _c01(vehicle_prox)
    mot = _c01(motion)
    pc = _c01(person_count / 5.0)
    return _c01(0.48 * vp + 0.32 * mot + 0.2 * pc)


@dataclass
class FrameSignals:
    person_count: int
    person_boxes: List[Tuple[float, float, float, float]]
    bag_boxes: List[Tuple[float, float, float, float]]
    motion_score: float
    proximity_cluster_score: float
    abandoned_max_seconds: float
    weapon_like_score: float
    weapon_boxes: List[Tuple[float, float, float, float]]
    fight_like_score: float
    fight_boxes: List[Tuple[float, float, float, float]]
    theft_like_score: float
    theft_boxes: List[Tuple[float, float, float, float]]
    missile_like_score: float
    missile_boxes: List[Tuple[float, float, float, float]]
    drone_like_score: float
    drone_boxes: List[Tuple[float, float, float, float]]
    accident_like_score: float = 0.0
    accident_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    vehicle_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    person_box_tags: List[str] = field(default_factory=list)
    bag_box_tags: List[str] = field(default_factory=list)
    weapon_box_tags: List[str] = field(default_factory=list)
    fight_box_tags: List[str] = field(default_factory=list)
    theft_box_tags: List[str] = field(default_factory=list)
    missile_box_tags: List[str] = field(default_factory=list)
    drone_box_tags: List[str] = field(default_factory=list)
    accident_box_tags: List[str] = field(default_factory=list)
    vehicle_box_tags: List[str] = field(default_factory=list)


def _centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class SimpleBagTracker:
    """Assign bag detections to track ids; estimate idle time and owner distance."""

    def __init__(
        self,
        fps: float,
        match_px: float = 80.0,
        owner_px: float = 120.0,
    ) -> None:
        self.fps = max(fps, 1e-3)
        self.match_px = match_px
        self.owner_px = owner_px
        self._next_id = 1
        self.tracks: Dict[int, Dict] = {}

    def update(
        self,
        bag_boxes: List[Tuple[float, float, float, float]],
        person_boxes: List[Tuple[float, float, float, float]],
    ) -> float:
        """Returns max abandoned duration (seconds) across tracks this frame."""
        used = set()
        # Match each detection to nearest existing centroid
        dets = [(i, _centroid(b)) for i, b in enumerate(bag_boxes)]
        for tid, tr in list(self.tracks.items()):
            tc = tr["centroid"]
            best = None
            best_d = 1e9
            for idx, c in dets:
                if idx in used:
                    continue
                d = _dist(tc, c)
                if d < best_d:
                    best_d = d
                    best = idx
            if best is not None and best_d < self.match_px:
                used.add(best)
                box = bag_boxes[best]
                tr["centroid"] = _centroid(box)
                tr["box"] = box
                tr["miss"] = 0
            else:
                tr["miss"] = tr.get("miss", 0) + 1

        for idx, box in enumerate(bag_boxes):
            if idx in used:
                continue
            cid = self._next_id
            self._next_id += 1
            self.tracks[cid] = {
                "centroid": _centroid(box),
                "box": box,
                "miss": 0,
                "stationary": 0.0,
                "last_move": 0.0,
            }

        # Remove stale
        dead = [tid for tid, tr in self.tracks.items() if tr["miss"] > 8]
        for tid in dead:
            del self.tracks[tid]

        max_abandon = 0.0
        for tid, tr in self.tracks.items():
            box = tr["box"]
            bc = _centroid(box)
            nearest_person = min(
                (_dist(bc, _centroid(pb)) for pb in person_boxes),
                default=1e9,
            )
            moved = False
            if "prev_c" in tr:
                if _dist(tr["prev_c"], bc) > 5.0:
                    moved = True
            tr["prev_c"] = bc
            if moved:
                tr["stationary"] = 0.0
            else:
                tr["stationary"] += 1.0 / self.fps

            if nearest_person < self.owner_px:
                tr["stationary"] = 0.0

            if tr["stationary"] > 0 and nearest_person > self.owner_px:
                max_abandon = max(max_abandon, tr["stationary"])

        return max_abandon


def proximity_score(
    person_boxes: List[Tuple[float, float, float, float]],
    frame_shape: Optional[Tuple[int, ...]] = None,
) -> float:
    """0-1 rough score: many close pairs -> higher. Threshold scales with frame size (HD road CCTV)."""
    if len(person_boxes) < 2:
        return 0.0
    cents = [_centroid(b) for b in person_boxes]
    close = 0
    pairs = 0
    if frame_shape is not None and len(frame_shape) >= 2:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        thr = float(max(52.0, min(0.13 * min(h, w), 240.0)))
    else:
        thr = 90.0
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            pairs += 1
            if _dist(cents[i], cents[j]) < thr:
                close += 1
    if pairs == 0:
        return 0.0
    base = float(min(1.0, (close / pairs) * 3.0))
    # Wide road / mela shots: many people may be pairwise "far" in pixels but still a gathering.
    n = len(person_boxes)
    if n >= 4:
        headcount_boost = min(1.0, (n - 3) * 0.09)
        base = min(1.0, max(base, headcount_boost * 0.85))
    return base


def motion_from_flow(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    mask_persons: Optional[np.ndarray] = None,
) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    if mask_persons is not None:
        m = mask_persons > 0
        if m.any():
            v = mag[m]
        else:
            v = mag.reshape(-1)
    else:
        v = mag.reshape(-1)
    # robust normalize
    p95 = float(np.percentile(v, 95)) + 1e-6
    score = float(np.clip(np.mean(v) / (p95 * 0.35), 0.0, 1.0))
    return score


DEFAULT_MISSILE_CLASS_NAMES = frozenset(
    {
        "missile",
        "missiles",
        "rocket",
        "projectile",
        "cruise_missile",
        "ballistic_missile",
        "rocket_motor",
    }
)

DEFAULT_DRONE_CLASS_NAMES = frozenset(
    {
        "drone",
        "uav",
        "quadcopter",
        "multicopter",
        "unmanned_aerial_vehicle",
        "fpv_drone",
        "drone_uav",
    }
)

# COCO-only proxies; add gun/knife via custom-trained weights + weapon_class_names.
DEFAULT_COCO_WEAPON_NAMES = frozenset({"baseball bat", "tennis racket"})

DEFAULT_FIGHT_CLASS_NAMES = frozenset(
    {"fight", "fighting", "punch", "violence", "brawl", "assault"}
)

DEFAULT_THEFT_CLASS_NAMES = frozenset(
    {"shoplifting", "stealing", "theft", "burglary", "robbery", "pickpocket"}
)

# COCO road / rail vehicles — used only for a weak "multi-vehicle scene" heuristic.
DEFAULT_COCO_VEHICLE_NAMES = frozenset(
    {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "train",
    }
)

# Custom weights must define these (or a subset) for real accident/crash boxes.
DEFAULT_ACCIDENT_CLASS_NAMES = frozenset(
    {
        "accident",
        "car_accident",
        "road_accident",
        "crash",
        "vehicle_crash",
        "collision",
        "wreck",
        "overturned_vehicle",
        "damaged_vehicle",
        "traffic_accident",
    }
)


class VideoAnalyzer:
    """
    Default yolov8n.pt (COCO) does **not** detect missiles or drones reliably.
    Use custom-trained weights (`best.pt`) with your class names; keep labels disjoint
    between missile vs drone sets so each box maps to one category.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: Optional[str] = None,
        missile_class_names: Optional[set[str]] = None,
        missile_conf_threshold: float = 0.4,
        drone_class_names: Optional[set[str]] = None,
        drone_conf_threshold: float = 0.4,
        weapon_class_names: Optional[set[str]] = None,
        weapon_conf_threshold: float = 0.35,
        fight_class_names: Optional[set[str]] = None,
        fight_conf_threshold: float = 0.4,
        theft_class_names: Optional[set[str]] = None,
        theft_conf_threshold: float = 0.4,
        accident_class_names: Optional[set[str]] = None,
        accident_conf_threshold: float = 0.4,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.missile_conf_threshold = missile_conf_threshold
        self.missile_class_names = {
            n.strip().lower()
            for n in (missile_class_names or DEFAULT_MISSILE_CLASS_NAMES)
        }
        self.drone_conf_threshold = drone_conf_threshold
        self.drone_class_names = {
            n.strip().lower()
            for n in (drone_class_names or DEFAULT_DRONE_CLASS_NAMES)
        }
        extra_w = {n.strip().lower() for n in (weapon_class_names or set()) if n.strip()}
        self.weapon_class_names = DEFAULT_COCO_WEAPON_NAMES | extra_w
        self.weapon_conf_threshold = weapon_conf_threshold
        self.fight_class_names = {
            n.strip().lower()
            for n in (fight_class_names or DEFAULT_FIGHT_CLASS_NAMES)
        }
        self.fight_conf_threshold = fight_conf_threshold
        self.theft_class_names = {
            n.strip().lower()
            for n in (theft_class_names or DEFAULT_THEFT_CLASS_NAMES)
        }
        self.theft_conf_threshold = theft_conf_threshold
        self.accident_class_names = {
            n.strip().lower()
            for n in (accident_class_names or DEFAULT_ACCIDENT_CLASS_NAMES)
        }
        self.accident_conf_threshold = accident_conf_threshold
        self.names = self.model.names
        self.class_to_name = {int(k): str(v) for k, v in self.names.items()}

    def _extract_boxes(
        self, result
    ) -> Tuple[
        List[Tuple[float, float, float, float]],
        List[str],
        List[Tuple[float, float, float, float]],
        List[str],
        List[Tuple[float, float, float, float]],
        List[str],
        List[Tuple[float, float, float, float]],
        List[str],
        List[Tuple[float, float, float, float]],
        List[str],
        List[Tuple[float, float, float, float]],
        List[str],
        List[Tuple[float, float, float, float]],
        List[str],
        List[Tuple[float, float, float, float]],
        List[str],
        List[Tuple[float, float, float, float]],
        List[str],
    ]:
        def _tag(n: str) -> str:
            t = n.strip().lower().replace(" ", "_")
            return t if t else "obj"

        person_boxes: List[Tuple[float, float, float, float]] = []
        person_tags: List[str] = []
        bag_boxes: List[Tuple[float, float, float, float]] = []
        bag_tags: List[str] = []
        weapon_boxes: List[Tuple[float, float, float, float]] = []
        weapon_tags: List[str] = []
        fight_boxes: List[Tuple[float, float, float, float]] = []
        fight_tags: List[str] = []
        theft_boxes: List[Tuple[float, float, float, float]] = []
        theft_tags: List[str] = []
        accident_boxes: List[Tuple[float, float, float, float]] = []
        accident_tags: List[str] = []
        missile_boxes: List[Tuple[float, float, float, float]] = []
        missile_tags: List[str] = []
        drone_boxes: List[Tuple[float, float, float, float]] = []
        drone_tags: List[str] = []
        vehicle_boxes: List[Tuple[float, float, float, float]] = []
        vehicle_tags: List[str] = []
        if result.boxes is None or len(result.boxes) == 0:
            return (
                person_boxes,
                person_tags,
                bag_boxes,
                bag_tags,
                weapon_boxes,
                weapon_tags,
                fight_boxes,
                fight_tags,
                theft_boxes,
                theft_tags,
                accident_boxes,
                accident_tags,
                missile_boxes,
                missile_tags,
                drone_boxes,
                drone_tags,
                vehicle_boxes,
                vehicle_tags,
            )
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        for b, c, cf in zip(xyxy, cls, conf):
            if cf < 0.35:
                continue
            name = self.class_to_name.get(int(c), "").strip().lower()
            x1, y1, x2, y2 = map(float, b)
            box = (x1, y1, x2, y2)
            if name == "person":
                person_boxes.append(box)
                person_tags.append(_tag(name))
            elif name in ("backpack", "handbag", "suitcase"):
                bag_boxes.append(box)
                bag_tags.append(_tag(name))
            elif name in self.weapon_class_names and cf >= self.weapon_conf_threshold:
                weapon_boxes.append(box)
                weapon_tags.append(_tag(name))
            elif name in self.fight_class_names and cf >= self.fight_conf_threshold:
                fight_boxes.append(box)
                fight_tags.append(_tag(name))
            elif name in self.theft_class_names and cf >= self.theft_conf_threshold:
                theft_boxes.append(box)
                theft_tags.append(_tag(name))
            elif name in self.accident_class_names and cf >= self.accident_conf_threshold:
                accident_boxes.append(box)
                accident_tags.append(_tag(name))
            elif name in self.missile_class_names and cf >= self.missile_conf_threshold:
                missile_boxes.append(box)
                missile_tags.append(_tag(name))
            elif name in self.drone_class_names and cf >= self.drone_conf_threshold:
                drone_boxes.append(box)
                drone_tags.append(_tag(name))
            elif name in DEFAULT_COCO_VEHICLE_NAMES:
                vehicle_boxes.append(box)
                vehicle_tags.append(_tag(name))
        return (
            person_boxes,
            person_tags,
            bag_boxes,
            bag_tags,
            weapon_boxes,
            weapon_tags,
            fight_boxes,
            fight_tags,
            theft_boxes,
            theft_tags,
            accident_boxes,
            accident_tags,
            missile_boxes,
            missile_tags,
            drone_boxes,
            drone_tags,
            vehicle_boxes,
            vehicle_tags,
        )

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        prev_gray: Optional[np.ndarray],
        gray: np.ndarray,
        tracker: SimpleBagTracker,
    ) -> FrameSignals:
        res = self.model.predict(
            frame_bgr, verbose=False, device=self.device, imgsz=640
        )[0]
        (
            person_boxes,
            person_tags,
            bag_boxes,
            bag_tags,
            weapon_boxes,
            weapon_tags,
            fight_boxes,
            fight_tags,
            theft_boxes,
            theft_tags,
            accident_boxes,
            accident_tags,
            missile_boxes,
            missile_tags,
            drone_boxes,
            drone_tags,
            vehicle_boxes,
            vehicle_tags,
        ) = self._extract_boxes(res)

        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        for pb in person_boxes:
            x1, y1, x2, y2 = map(int, pb)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        motion_score = 0.0
        if prev_gray is not None:
            motion_score = motion_from_flow(prev_gray, gray, mask)

        abandoned_sec = tracker.update(bag_boxes, person_boxes)
        prox = proximity_score(person_boxes, frame_bgr.shape)
        weapon_score = min(1.0, len(weapon_boxes) * 0.4)
        missile_score = min(1.0, len(missile_boxes) * 0.55)
        drone_score = min(1.0, len(drone_boxes) * 0.5)

        fh = fight_heuristic_score(motion_score, prox, len(person_boxes))
        fight_det = min(1.0, len(fight_boxes) * 0.52)
        fight_score = max(fh, fight_det)

        th = theft_heuristic_score(
            abandoned_sec, motion_score, len(person_boxes), len(bag_boxes)
        )
        theft_det = min(1.0, len(theft_boxes) * 0.55)
        theft_score = max(th, theft_det)

        v_prox = vehicle_proximity_score(vehicle_boxes)
        ah = accident_heuristic_score(
            len(vehicle_boxes), v_prox, motion_score, len(person_boxes)
        )
        accident_det = min(1.0, len(accident_boxes) * 0.55)
        accident_score = max(ah, accident_det)

        return FrameSignals(
            person_count=len(person_boxes),
            person_boxes=person_boxes,
            bag_boxes=bag_boxes,
            motion_score=motion_score,
            proximity_cluster_score=prox,
            abandoned_max_seconds=abandoned_sec,
            weapon_like_score=weapon_score,
            weapon_boxes=weapon_boxes,
            fight_like_score=fight_score,
            fight_boxes=fight_boxes,
            theft_like_score=theft_score,
            theft_boxes=theft_boxes,
            missile_like_score=missile_score,
            missile_boxes=missile_boxes,
            drone_like_score=drone_score,
            drone_boxes=drone_boxes,
            accident_like_score=accident_score,
            accident_boxes=accident_boxes,
            vehicle_boxes=vehicle_boxes,
            person_box_tags=person_tags,
            bag_box_tags=bag_tags,
            weapon_box_tags=weapon_tags,
            fight_box_tags=fight_tags,
            theft_box_tags=theft_tags,
            missile_box_tags=missile_tags,
            drone_box_tags=drone_tags,
            accident_box_tags=accident_tags,
            vehicle_box_tags=vehicle_tags,
        )


def _bracket(inner: str) -> str:
    return f"[{inner}]"


def _draw_tagged_box(
    out: np.ndarray,
    box: Tuple[float, float, float, float],
    tag_inner: str,
    rect_bgr: Tuple[int, int, int],
    label_bg_bgr: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img = out.shape[:2]
    cv2.rectangle(out, (x1, y1), (x2, y2), rect_bgr, thickness)
    text = _bracket(tag_inner)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    tthick = 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, tthick)
    pad = 3
    tx = max(0, min(x1, w_img - tw - 2 * pad - 1))
    ty = max(th + pad + 4, y1 - 3)
    y_top = ty - th - pad
    y_bot = min(h_img - 1, ty + bl + pad)
    x_r = min(w_img - 1, tx + tw + 2 * pad)
    cv2.rectangle(out, (tx, y_top), (x_r, y_bot), label_bg_bgr, -1)
    cv2.putText(
        out,
        text,
        (tx + pad, ty),
        font,
        scale,
        (255, 255, 255),
        tthick,
        cv2.LINE_AA,
    )


def _tag_for(boxes: List[Tuple[float, float, float, float]], tags: List[str], i: int, fb: str) -> str:
    if i < len(tags) and tags[i]:
        return tags[i]
    return fb


def draw_overlay(
    frame_bgr: np.ndarray,
    signals: FrameSignals,
    risk_total: float,
    labels: List[str],
    location_line: Optional[str] = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    h_img, w_img = out.shape[:2]

    # Person = green box; brighter / thicker when this frame has fused risk flags
    risk_emphasis = bool(labels)
    p_bgr = (0, 255, 0) if risk_emphasis else (0, 235, 0)
    p_bg = (0, 140, 0) if risk_emphasis else (0, 110, 0)
    p_thick = 5 if risk_emphasis else 3
    for i, pb in enumerate(signals.person_boxes):
        _draw_tagged_box(
            out,
            pb,
            _tag_for(signals.person_boxes, signals.person_box_tags, i, "person"),
            p_bgr,
            p_bg,
            thickness=p_thick,
        )
    for i, bb in enumerate(signals.bag_boxes):
        _draw_tagged_box(
            out,
            bb,
            _tag_for(signals.bag_boxes, signals.bag_box_tags, i, "bag"),
            (0, 165, 255),
            (0, 90, 140),
        )
    for i, wb in enumerate(signals.weapon_boxes):
        _draw_tagged_box(
            out,
            wb,
            _tag_for(signals.weapon_boxes, signals.weapon_box_tags, i, "weapon"),
            (0, 0, 255),
            (0, 0, 120),
            thickness=3,
        )
    for i, fb in enumerate(signals.fight_boxes):
        _draw_tagged_box(
            out,
            fb,
            _tag_for(signals.fight_boxes, signals.fight_box_tags, i, "fight"),
            (0, 140, 255),
            (0, 80, 160),
            thickness=2,
        )
    for i, tb in enumerate(signals.theft_boxes):
        _draw_tagged_box(
            out,
            tb,
            _tag_for(signals.theft_boxes, signals.theft_box_tags, i, "theft"),
            (255, 200, 0),
            (120, 100, 0),
        )
    for i, mb in enumerate(signals.missile_boxes):
        _draw_tagged_box(
            out,
            mb,
            _tag_for(signals.missile_boxes, signals.missile_box_tags, i, "missile"),
            (0, 0, 255),
            (0, 0, 100),
        )
    for i, db in enumerate(signals.drone_boxes):
        _draw_tagged_box(
            out,
            db,
            _tag_for(signals.drone_boxes, signals.drone_box_tags, i, "drone"),
            (255, 0, 255),
            (120, 0, 120),
        )
    for i, ab in enumerate(signals.accident_boxes):
        _draw_tagged_box(
            out,
            ab,
            _tag_for(signals.accident_boxes, signals.accident_box_tags, i, "accident"),
            (0, 100, 255),
            (0, 60, 160),
            thickness=3,
        )
    if signals.accident_like_score >= 0.28 and signals.vehicle_boxes:
        for i, vb in enumerate(signals.vehicle_boxes):
            _draw_tagged_box(
                out,
                vb,
                _tag_for(signals.vehicle_boxes, signals.vehicle_box_tags, i, "vehicle"),
                (200, 200, 200),
                (80, 80, 80),
                thickness=1,
            )

    metrics_h = 52 if location_line else 38
    risk_band_h = 36
    txt = (
        f"Risk:{risk_total:.0f} Acc:{signals.accident_like_score:.2f} P:{signals.person_count} "
        f"Mot:{signals.motion_score:.2f} Fgt:{signals.fight_like_score:.2f} "
        f"Tft:{signals.theft_like_score:.2f}"
    )
    cv2.rectangle(out, (0, 0), (w_img, metrics_h), (0, 0, 0), -1)
    cv2.putText(
        out,
        txt[:125],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if location_line:
        cv2.putText(
            out,
            location_line[:140],
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 255, 200),
            1,
            cv2.LINE_AA,
        )

    # Green band = fused risk types (square-bracket labels)
    y_r0 = metrics_h
    y_r1 = metrics_h + risk_band_h
    cv2.rectangle(out, (0, y_r0), (w_img, y_r1), (0, 160, 0), -1)
    cv2.rectangle(out, (0, y_r0), (w_img, y_r1), (0, 220, 100), 2)
    if labels:
        risk_line = "RISK FLAGS: " + " ".join(_bracket(lb) for lb in labels)
    else:
        risk_line = "RISK FLAGS: " + _bracket("no_fused_alert") + " " + _bracket("see_detections_above")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 1
    max_tw = w_img - 16
    while scale >= 0.3:
        (tw, th), bl = cv2.getTextSize(risk_line, font, scale, thick)
        if tw <= max_tw:
            break
        scale -= 0.04
    else:
        risk_line = risk_line[:72] + "..."
        (tw, th), bl = cv2.getTextSize(risk_line, font, 0.3, thick)
    ty = y_r0 + int((risk_band_h + th) / 2) - 2
    cv2.putText(
        out,
        risk_line,
        (8, min(y_r1 - 6, ty)),
        font,
        scale,
        (255, 255, 255),
        thick,
        cv2.LINE_AA,
    )

    foot = (
        "Green = person; orange-red = accident/crash class; gray = vehicles when accident-hint is raised. "
        "Acc score mixes custom crash boxes + weak multi-vehicle cue — not proof of a crash."
    )
    fs = 0.42
    (fw, fh), _ = cv2.getTextSize(foot[:90], font, fs, 1)
    y_f = h_img - 8
    cv2.rectangle(out, (0, y_f - fh - 8), (w_img, h_img), (0, 0, 0), -1)
    cv2.putText(
        out,
        foot[: min(120, len(foot))],
        (6, y_f - 4),
        font,
        fs,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return out
