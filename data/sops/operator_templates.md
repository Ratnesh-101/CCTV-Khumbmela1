# Operator message templates

## SMS / Telegram short form
- FIGHT: `[ALERT] Possible disturbance Cam {cam} score={score} t={t}. Verify live feed.`
- BAG: `[ALERT] Unattended object Cam {cam} idle={sec}s score={score}.`
- CROWD: `[ALERT] Crowd/motion anomaly Cam {cam} people={n} score={score}.`
- ACOUSTIC: `[ALERT] Loud transient (unverified) zone={zone} score={score}.`
- HELP: `[HELP] User requested assistance lat={lat} lng={lng} acc={acc}m note={note}.`

## Public web banner (non-panic wording)
- VERIFY: `Unverified safety signal near {zone}. Follow staff instructions.`
- CRITICAL: `Confirmed security incident — follow venue staff directions immediately.`

## Missile-like (custom model, unverified)
- OP: `[ALERT] missile_like_unverified Cam {cam} score={score} boxes={n}. Confirm visually; follow official air-raid channels.`

## Drone / UAV (custom model, unverified)
- OP: `[ALERT] unauthorized_drone_unverified Cam {cam} score={score} drones={n}. Verify vs allowlist / geofence; do not assume hostile intent.`

## Mass gathering (RED — auto-escalate)
- OP: `[RED ALERT] mass_gathering Cam {cam} people={n} score={score} t={t}. Dispatch crowd team; open exits.`
- PUBLIC: `Crowd management in progress near {zone}. Please follow staff directions and use marked exit routes.`

## Pocket knife / blade (RED)
- OP: `[RED ALERT] pocket_knife_detected Cam {cam} score={score} t={t}. Confirm visually; dispatch response team.`

## Gun pointing (RED — immediate)
- OP: `[RED ALERT] gun_pointing_detected Cam {cam} score={score} t={t}. Armed response + law enforcement NOW. Confirm before public broadcast.`

## Unauthorised thela (RED)
- OP: `[RED ALERT] unauthorised_thela Cam {cam} score={score} t={t}. Verify and request relocation; escalate to police if non-compliant.`

## Bad road / road hazard (RED)
- OP: `[RED ALERT] bad_road_hazard Cam {cam} score={score} t={t}. Notify traffic management; check for affected persons/vehicles.`
