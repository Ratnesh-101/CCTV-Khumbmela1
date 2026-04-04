"""
Transparent risk scoring from CV / audio heuristic signals.
Outputs 0-100 score with per-component breakdown (for pandas logs & charts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RiskComponents:
    crowd_density: float = 0.0  # 0-1 normalized
    motion_anomaly: float = 0.0  # generic motion / proximity 0-1
    abandoned_object: float = 0.0  # 0-1
    acoustic_threat: float = 0.0  # 0-1 optional
    weapon_like: float = 0.0  # YOLO + COCO proxies — unverified
    missile_like: float = 0.0  # custom YOLO
    drone_like: float = 0.0  # custom YOLO
    wifi_rf_anomaly: float = 0.0  # Wi-Fi IF
    # Heuristic + optional custom classes (fight, theft) — demo only, not court-grade.
    fight_like: float = 0.0
    theft_like: float = 0.0
    accident_like: float = 0.0  # custom "crash" classes + vehicle heuristic (demo)


@dataclass
class RiskScore:
    total: float
    components: RiskComponents
    labels: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "crowd_density": self.components.crowd_density,
            "motion_anomaly": self.components.motion_anomaly,
            "abandoned_object": self.components.abandoned_object,
            "acoustic_threat": self.components.acoustic_threat,
            "weapon_like": self.components.weapon_like,
            "missile_like": self.components.missile_like,
            "drone_like": self.components.drone_like,
            "wifi_rf_anomaly": self.components.wifi_rf_anomaly,
            "fight_like": self.components.fight_like,
            "theft_like": self.components.theft_like,
            "accident_like": self.components.accident_like,
            "labels": ",".join(self.labels),
        }


# Weights (must sum to 1.0)
WEIGHTS = {
    "crowd_density": 0.05,
    "motion_anomaly": 0.10,
    "abandoned_object": 0.09,
    "acoustic_threat": 0.05,
    "weapon_like": 0.09,
    "missile_like": 0.08,
    "drone_like": 0.08,
    "wifi_rf_anomaly": 0.07,
    "fight_like": 0.14,
    "theft_like": 0.13,
    "accident_like": 0.12,
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_risk(
    *,
    person_count: int,
    person_count_baseline: float = 6.0,
    motion_score: float = 0.0,
    proximity_cluster_score: float = 0.0,
    abandoned_seconds: float = 0.0,
    abandoned_threshold_sec: float = 45.0,
    acoustic_score: float = 0.0,
    weapon_like_score: float = 0.0,
    missile_like_score: float = 0.0,
    drone_like_score: float = 0.0,
    wifi_rf_anomaly_score: float = 0.0,
    fight_like_score: float = 0.0,
    theft_like_score: float = 0.0,
    accident_like_score: float = 0.0,
    vehicle_count: int = 0,
) -> RiskScore:
    """
    motion_score: optical-flow energy in person ROIs, 0-1.
    proximity_cluster_score: 0-1 when multiple persons are close.
    fight_like_score: combined fight heuristic + optional YOLO "fight" classes.
    theft_like_score: unattended-bag heuristic + optional YOLO theft/shoplifting classes.
    accident_like_score: custom crash/accident detections + weak multi-vehicle heuristic (not a real crash detector).
    vehicle_count: COCO-style vehicles in frame — with people, hints road / junction mass gathering.
    """
    crowd_density = _clip01(person_count / max(person_count_baseline, 1.0))
    gathering_signal = _clip01(
        0.48 * crowd_density + 0.52 * _clip01(proximity_cluster_score)
    )
    if person_count >= 4:
        gathering_signal = max(
            gathering_signal,
            _clip01((person_count - 3) * 0.095),
        )
    motion_anomaly = _clip01(0.55 * motion_score + 0.45 * proximity_cluster_score)
    abandoned_object = _clip01(abandoned_seconds / max(abandoned_threshold_sec, 1.0))
    acoustic_threat = _clip01(acoustic_score)
    weapon_like = _clip01(weapon_like_score)
    missile_like = _clip01(missile_like_score)
    drone_like = _clip01(drone_like_score)
    wifi_rf_anomaly = _clip01(wifi_rf_anomaly_score)
    fight_like = _clip01(fight_like_score)
    theft_like = _clip01(theft_like_score)
    accident_like = _clip01(accident_like_score)

    comps = RiskComponents(
        crowd_density=crowd_density,
        motion_anomaly=motion_anomaly,
        abandoned_object=abandoned_object,
        acoustic_threat=acoustic_threat,
        weapon_like=weapon_like,
        missile_like=missile_like,
        drone_like=drone_like,
        wifi_rf_anomaly=wifi_rf_anomaly,
        fight_like=fight_like,
        theft_like=theft_like,
        accident_like=accident_like,
    )

    total = (
        WEIGHTS["crowd_density"] * crowd_density
        + WEIGHTS["motion_anomaly"] * motion_anomaly
        + WEIGHTS["abandoned_object"] * abandoned_object
        + WEIGHTS["acoustic_threat"] * acoustic_threat
        + WEIGHTS["weapon_like"] * weapon_like
        + WEIGHTS["missile_like"] * missile_like
        + WEIGHTS["drone_like"] * drone_like
        + WEIGHTS["wifi_rf_anomaly"] * wifi_rf_anomaly
        + WEIGHTS["fight_like"] * fight_like
        + WEIGHTS["theft_like"] * theft_like
        + WEIGHTS["accident_like"] * accident_like
    ) * 100.0

    labels: List[str] = []
    if motion_anomaly > 0.55:
        labels.append("disturbance_motion")
    if abandoned_object > 0.5:
        labels.append("unattended_object")
    if person_count >= 3 and gathering_signal >= 0.24:
        labels.append("mass_gathering")
    if crowd_density >= 0.55 or (
        person_count >= 4 and gathering_signal >= 0.38
    ):
        labels.append("dense_crowd")
    if (
        vehicle_count >= 1
        and person_count >= 3
        and gathering_signal >= 0.22
    ):
        labels.append("road_gathering_vehicles")
    if acoustic_threat > 0.6:
        labels.append("loud_transient")
    if weapon_like > 0.5:
        labels.append("weapon_like_unverified")
    if missile_like > 0.35:
        labels.append("missile_like_unverified")
    if drone_like > 0.35:
        labels.append("unauthorized_drone_unverified")
    if wifi_rf_anomaly > 0.45:
        labels.append("wifi_drone_anomaly_unverified")
    if fight_like > 0.5:
        labels.append("fight_like_unverified")
    if theft_like > 0.45:
        labels.append("theft_like_unverified")
    if accident_like > 0.48:
        labels.append("road_accident_hint_unverified")
    elif accident_like > 0.32:
        labels.append("traffic_incident_heuristic")

    return RiskScore(
        total=float(round(_clip01(total / 100.0) * 100.0, 2)),
        components=comps,
        labels=labels,
        raw={
            "person_count": person_count,
            "motion_score": motion_score,
            "proximity_cluster_score": proximity_cluster_score,
            "gathering_signal": gathering_signal,
            "vehicle_count": vehicle_count,
            "abandoned_seconds": abandoned_seconds,
            "wifi_rf_anomaly": wifi_rf_anomaly,
            "fight_like": fight_like,
            "theft_like": theft_like,
            "accident_like": accident_like,
        },
    )


def risk_bucket(score: float) -> int:
    """0..9 discrete bucket for RL state."""
    return int(_clip01(score / 100.0) * 9.999)
