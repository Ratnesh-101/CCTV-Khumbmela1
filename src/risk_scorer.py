"""
Transparent risk scoring from CV / audio heuristic signals.
Outputs 0-100 score with per-component breakdown (for pandas logs & charts).

Changes (v2):
- mass_gathering is now a RED alert (score forced to >= 80 when triggered)
- New threat categories: pocket_knife, gun_pointing, unauthorised_thela, bad_road
- Raised weight on gathering/crowd signals
- Tighter label thresholds for higher precision
- Confidence scoring improved throughout
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RiskComponents:
    crowd_density: float = 0.0
    motion_anomaly: float = 0.0
    abandoned_object: float = 0.0
    acoustic_threat: float = 0.0
    weapon_like: float = 0.0
    missile_like: float = 0.0
    drone_like: float = 0.0
    wifi_rf_anomaly: float = 0.0
    fight_like: float = 0.0
    theft_like: float = 0.0
    accident_like: float = 0.0
    # --- NEW CATEGORIES ---
    gathering_risk: float = 0.0        # Mass gathering (now explicit + RED)
    pocket_knife_like: float = 0.0     # Pocket knife / small blade detection
    gun_pointing_like: float = 0.0     # Gun aimed / gun-pointing posture
    unauthorised_thela: float = 0.0    # Unauthorised street vendor / thela
    bad_road: float = 0.0              # Road hazard / pothole / debris


@dataclass
class RiskScore:
    total: float
    components: RiskComponents
    labels: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    alert_level: str = "GREEN"   # GREEN / AMBER / RED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "alert_level": self.alert_level,
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
            "gathering_risk": self.components.gathering_risk,
            "pocket_knife_like": self.components.pocket_knife_like,
            "gun_pointing_like": self.components.gun_pointing_like,
            "unauthorised_thela": self.components.unauthorised_thela,
            "bad_road": self.components.bad_road,
            "labels": ",".join(self.labels),
        }


# ---------------------------------------------------------------------------
# Weights — must sum to 1.0
# ---------------------------------------------------------------------------
WEIGHTS = {
    "crowd_density":       0.04,
    "motion_anomaly":      0.08,
    "abandoned_object":    0.07,
    "acoustic_threat":     0.04,
    "weapon_like":         0.07,
    "missile_like":        0.05,
    "drone_like":          0.05,
    "wifi_rf_anomaly":     0.04,
    "fight_like":          0.09,
    "theft_like":          0.08,
    "accident_like":       0.07,
    # new
    "gathering_risk":      0.09,   # explicit + RED
    "pocket_knife_like":   0.07,
    "gun_pointing_like":   0.09,   # highest immediate-danger
    "unauthorised_thela":  0.04,
    "bad_road":            0.07,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, f"Weights sum={sum(WEIGHTS.values())}"

# ---------------------------------------------------------------------------
# RED alert — any matching condition forces alert_level=RED and score >= 80
# ---------------------------------------------------------------------------
RED_FLOOR = 80.0

RED_CONDITIONS = {
    "mass_gathering_risk":   lambda c, _lbs: c.gathering_risk >= 0.55,
    "gun_pointing_detected": lambda c, _lbs: c.gun_pointing_like >= 0.40,
    "pocket_knife_detected": lambda c, _lbs: c.pocket_knife_like >= 0.45,
    "unauthorised_thela":    lambda c, _lbs: c.unauthorised_thela >= 0.50,
    "bad_road_hazard":       lambda c, _lbs: c.bad_road >= 0.50,
    "weapon_like_high":      lambda c, _lbs: c.weapon_like >= 0.55,
    "fight_confirmed":       lambda c, _lbs: c.fight_like >= 0.65,
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
    # NEW
    pocket_knife_score: float = 0.0,
    gun_pointing_score: float = 0.0,
    unauthorised_thela_score: float = 0.0,
    bad_road_score: float = 0.0,
) -> RiskScore:
    crowd_density = _clip01(person_count / max(person_count_baseline, 1.0))

    gathering_signal = _clip01(
        0.45 * crowd_density + 0.55 * _clip01(proximity_cluster_score)
    )
    if person_count >= 4:
        gathering_signal = max(
            gathering_signal,
            _clip01((person_count - 3) * 0.12),
        )
    gathering_risk = gathering_signal

    motion_anomaly    = _clip01(0.55 * motion_score + 0.45 * proximity_cluster_score)
    abandoned_object  = _clip01(abandoned_seconds / max(abandoned_threshold_sec, 1.0))
    acoustic_threat   = _clip01(acoustic_score)
    weapon_like       = _clip01(weapon_like_score)
    missile_like      = _clip01(missile_like_score)
    drone_like        = _clip01(drone_like_score)
    wifi_rf_anomaly   = _clip01(wifi_rf_anomaly_score)
    fight_like        = _clip01(fight_like_score)
    theft_like        = _clip01(theft_like_score)
    accident_like     = _clip01(accident_like_score)
    pocket_knife_like = _clip01(pocket_knife_score)
    gun_pointing_like = _clip01(gun_pointing_score)
    unauthorised_thela = _clip01(unauthorised_thela_score)
    bad_road          = _clip01(bad_road_score)

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
        gathering_risk=gathering_risk,
        pocket_knife_like=pocket_knife_like,
        gun_pointing_like=gun_pointing_like,
        unauthorised_thela=unauthorised_thela,
        bad_road=bad_road,
    )

    total = (
        WEIGHTS["crowd_density"]       * crowd_density
        + WEIGHTS["motion_anomaly"]    * motion_anomaly
        + WEIGHTS["abandoned_object"]  * abandoned_object
        + WEIGHTS["acoustic_threat"]   * acoustic_threat
        + WEIGHTS["weapon_like"]       * weapon_like
        + WEIGHTS["missile_like"]      * missile_like
        + WEIGHTS["drone_like"]        * drone_like
        + WEIGHTS["wifi_rf_anomaly"]   * wifi_rf_anomaly
        + WEIGHTS["fight_like"]        * fight_like
        + WEIGHTS["theft_like"]        * theft_like
        + WEIGHTS["accident_like"]     * accident_like
        + WEIGHTS["gathering_risk"]    * gathering_risk
        + WEIGHTS["pocket_knife_like"] * pocket_knife_like
        + WEIGHTS["gun_pointing_like"] * gun_pointing_like
        + WEIGHTS["unauthorised_thela"]* unauthorised_thela
        + WEIGHTS["bad_road"]          * bad_road
    ) * 100.0

    # --- Labels (tighter thresholds for precision) ---
    labels: List[str] = []

    if motion_anomaly > 0.60:
        labels.append("disturbance_motion")
    if abandoned_object > 0.55:
        labels.append("unattended_object")
    if person_count >= 3 and gathering_risk >= 0.28:
        labels.append("mass_gathering")
    if crowd_density >= 0.60 or (person_count >= 4 and gathering_risk >= 0.42):
        labels.append("dense_crowd")
    if vehicle_count >= 1 and person_count >= 3 and gathering_risk >= 0.25:
        labels.append("road_gathering_vehicles")
    if acoustic_threat > 0.65:
        labels.append("loud_transient")
    if weapon_like > 0.55:
        labels.append("weapon_like_unverified")
    if missile_like > 0.40:
        labels.append("missile_like_unverified")
    if drone_like > 0.40:
        labels.append("unauthorized_drone_unverified")
    if wifi_rf_anomaly > 0.50:
        labels.append("wifi_drone_anomaly_unverified")
    if fight_like > 0.55:
        labels.append("fight_like_unverified")
    if theft_like > 0.50:
        labels.append("theft_like_unverified")
    if accident_like > 0.50:
        labels.append("road_accident_hint_unverified")
    elif accident_like > 0.35:
        labels.append("traffic_incident_heuristic")

    # NEW — all trigger RED
    if pocket_knife_like >= 0.45:
        labels.append("pocket_knife_detected")
    if gun_pointing_like >= 0.40:
        labels.append("gun_pointing_detected")
    if unauthorised_thela >= 0.50:
        labels.append("unauthorised_thela_detected")
    if bad_road >= 0.50:
        labels.append("bad_road_hazard")

    # --- Determine alert level ---
    is_red = any(fn(comps, labels) for fn in RED_CONDITIONS.values())

    if is_red:
        total = max(total, RED_FLOOR)
        alert_level = "RED"
    elif total >= 50:
        alert_level = "AMBER"
    else:
        alert_level = "GREEN"

    return RiskScore(
        total=float(round(_clip01(total / 100.0) * 100.0, 2)),
        components=comps,
        labels=labels,
        alert_level=alert_level,
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
            "gathering_risk": gathering_risk,
            "pocket_knife_like": pocket_knife_like,
            "gun_pointing_like": gun_pointing_like,
            "unauthorised_thela": unauthorised_thela,
            "bad_road": bad_road,
        },
    )


def risk_bucket(score: float) -> int:
    """0..9 discrete bucket for RL state."""
    return int(_clip01(score / 100.0) * 9.999)
