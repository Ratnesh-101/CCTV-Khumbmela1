"""
Browser-friendly video output. OpenCV's default mp4v (MPEG-4 Part 2) often fails in HTML5 players.
We try H.264-style fourcc first, then fall back to Motion JPEG in an .avi container.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2


def open_video_writer(
    out_path: Path,
    fps: float,
    width: int,
    height: int,
) -> Tuple[cv2.VideoWriter, Path]:
    """
    Returns (writer, path_to_use). path_to_use may differ from out_path if we fall back to .avi.
    Caller must release the writer and use path_to_use for playback / download.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(fps) if fps and fps > 1e-3 else 25.0
    size = (int(width), int(height))

    # (path_suffix, fourcc_string) — order matters: prefer H.264 in .mp4 for browsers
    attempts: list[tuple[Path, str]] = [
        (out_path.with_suffix(".mp4"), "avc1"),
        (out_path.with_suffix(".mp4"), "H264"),
        (out_path.with_suffix(".mp4"), "X264"),
        (out_path.with_suffix(".mp4"), "mp4v"),
        (out_path.with_suffix(".avi"), "MJPG"),
    ]

    for path, fc in attempts:
        fourcc = cv2.VideoWriter_fourcc(*fc)
        w = cv2.VideoWriter(str(path), fourcc, fps, size)
        if w.isOpened():
            return w, path
        w.release()

    raise OSError(
        "Could not open any OpenCV video writer for this machine. "
        "Try installing an OpenCV build with ffmpeg support, or use a shorter clip."
    )
