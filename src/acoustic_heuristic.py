"""
Very simple microphone / audio-file heuristic for impulsive transients (demo only).
Not a reliable gunshot detector — label outputs as "loud_transient".
"""
from __future__ import annotations

import numpy as np


def transient_score_from_waveform(
    samples: np.ndarray,
    sample_rate: int,
    window_ms: float = 50.0,
) -> float:
    """Return 0-1 score from raw mono float32/int16 waveform."""
    if samples is None or len(samples) < 16:
        return 0.0
    x = samples.astype(np.float32)
    if x.max() > 1.5:
        x = x / 32768.0
    n = len(x)
    win = max(1, int(sample_rate * (window_ms / 1000.0)))
    max_rms = 0.0
    for i in range(0, n - win, win):
        chunk = x[i : i + win]
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        max_rms = max(max_rms, rms)
    # crude map
    return float(np.clip(max_rms * 18.0, 0.0, 1.0))
