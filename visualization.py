"""Diagnostic panels, masks, tracks and stop-line event overlays."""

from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig
from .detector.pipeline import FrameResult


def draw_overlay(frame: np.ndarray, result: FrameResult, cfg: DetectorConfig) -> np.ndarray:
    """Show complete boxes and only currently observed track positions."""
    image = frame.copy()
    # The green tint is the actual cleaned, ROI-restricted foreground mask.
    selected = result.mask > 0
    image[selected] = (image[selected].astype(np.float32) * 0.55 + np.array([0, 255, 80]) * 0.45).astype(np.uint8)
    if cfg.roi_polygon:
        cv2.polylines(image, [np.asarray(cfg.roi_polygon, np.int32)], True, (255, 180, 0), 1)
    if cfg.line:
        cv2.arrowedLine(image, tuple(cfg.line[0]), tuple(cfg.line[1]), (0, 220, 255), 2, tipLength=0.06)
    for track in result.tracks:
        if track.missed:
            continue
        x, y, width, height = track.bbox
        cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (0, 220, 0), 1)
        cv2.circle(image, track.centroid, 2, (255, 255, 255), -1)
        cv2.putText(image, f"ID {track.track_id}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if len(track.history) > 1:
            cv2.polylines(image, [np.asarray(track.history, np.int32)], False, (255, 100, 100), 1)
    state = "WARMUP / RECOVERY" if result.settling else "ALARM" if result.alarm.active else "MONITORING"
    color = (0, 0, 255) if result.alarm.active else (255, 255, 255)
    cv2.rectangle(image, (0, 0), (image.shape[1], 39), (25, 25, 25), -1)
    cv2.putText(image, state, (7, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    label = f"{cfg.method} | objects {len(result.objects)} | crossings {result.total_crossings}"
    cv2.putText(image, label, (7, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (215, 215, 215), 1, cv2.LINE_AA)
    return image


def _panel(image: np.ndarray, title: str) -> np.ndarray:
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    height, width = result.shape[:2]
    cv2.rectangle(result, (0, max(0, height - 23)), (width, height), (20, 20, 20), -1)
    cv2.putText(result, title, (6, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 225, 255), 1, cv2.LINE_AA)
    return result


def make_diagnostic_view(frame: np.ndarray, result: FrameResult, cfg: DetectorConfig) -> np.ndarray:
    """Four full-resolution panels, producing exactly 2W x 2H pixels."""
    top = np.hstack([_panel(frame, "1. Original"), _panel(result.blurred, "2. Grayscale + blur")])
    bottom = np.hstack([_panel(result.raw_mask, "3. Foreground (raw)"),
                        _panel(draw_overlay(frame, result, cfg), "4. Clean mask + tracks + line")])
    return np.vstack([top, bottom])


def save_snapshot(image: np.ndarray, directory: Path, frame_index: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"frame_{frame_index:06d}.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save snapshot: {path}")
    return path
