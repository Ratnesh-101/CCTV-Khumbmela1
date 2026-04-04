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
