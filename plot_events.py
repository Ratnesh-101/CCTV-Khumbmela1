"""Pandas + Matplotlib: plot risk timeline from process_video output."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "output" / "events.csv"
if not csv_path.exists():
    raise SystemExit(f"Missing {csv_path} — run process_video.py first.")

df = pd.read_csv(csv_path)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["t_sec"], df["risk_total"], label="risk_total", color="#c0392b")
ax.fill_between(df["t_sec"], 0, df["risk_total"], alpha=0.12, color="#c0392b")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Risk (0-100)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = ROOT / "output" / "risk_matplotlib.png"
fig.savefig(out, dpi=150)
print("Saved", out)
