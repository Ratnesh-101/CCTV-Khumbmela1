"""Operator notifications: Telegram bot or webhook (optional)."""
from __future__ import annotations

import os
from typing import List, Optional

import requests


def send_telegram(text: str) -> tuple[bool, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(
        url,
        json={"chat_id": chat_id, "text": text[:3500]},
        timeout=15,
    )
    if r.ok:
        return True, "ok"
    return False, r.text[:500]


def send_slack_webhook(text: str) -> tuple[bool, str]:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False, "SLACK_WEBHOOK_URL not set"
    r = requests.post(url, json={"text": text[:3500]}, timeout=15)
    if r.ok:
        return True, "ok"
    return False, r.text[:500]


def format_alert_location(
    message: str,
    *,
    site_name: str = "",
    lat: Optional[float] = None,
    lng: Optional[float] = None,
) -> str:
    """Prefix operator alerts with fixed camera/site coordinates (demo: not device GPS)."""
    bits: List[str] = []
    if site_name.strip():
        bits.append(f"site={site_name.strip()}")
    if lat is not None and lng is not None:
        bits.append(f"gps={lat:.6f},{lng:.6f}")
    if not bits:
        return message
    return "[" + " ".join(bits) + "] " + message


def notify_operator(message: str) -> str:
    ok, msg = send_telegram(message)
    if ok:
        return "telegram:" + msg
    ok2, msg2 = send_slack_webhook(message)
    if ok2:
        return "slack:" + msg2
    return "log_only:" + msg
