"""Segmentation, frame, event and throughput metrics."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return pixel precision, recall, F1 and IoU for two binary masks."""
    if prediction.shape != target.shape:
        raise ValueError(f"mask shape mismatch: {prediction.shape} != {target.shape}")
    pred = prediction > 0
    truth = target > 0
    tp = np.count_nonzero(pred & truth)
    fp = np.count_nonzero(pred & ~truth)
    fn = np.count_nonzero(~pred & truth)
    union = np.count_nonzero(pred | truth)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / union if union else 1.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou)}


def event_metrics(event_times: list[float], expected_times: list[float] | None = None, duration_seconds: float = 0.0, tolerance_seconds: float = 1.0) -> dict[str, float]:
    """Match predicted events to expected times; omit false-alarm claims without labels."""
    result = {"events": float(len(event_times))}
    if expected_times is None:
        return result
    expected = sorted(expected_times)
    unmatched = list(event_times)
    matched = 0
    delays: list[float] = []
    for target in expected:
        candidates = [value for value in unmatched if abs(value - target) <= tolerance_seconds]
        if candidates:
            value = min(candidates, key=lambda candidate: abs(candidate - target))
            unmatched.remove(value)
            matched += 1
            delays.append(value - target)
    false_alarms = len(unmatched)
    hours = duration_seconds / 3600 if duration_seconds > 0 else 0
    result.update({
        "expected_events": float(len(expected)),
        "matched_events": float(matched),
        "false_alarms": float(false_alarms),
        "false_alarms_per_hour": false_alarms / hours if hours else float(false_alarms),
        "mean_delay_seconds": float(np.mean(delays)) if delays else 0.0,
    })
    return result


@dataclass
class ThroughputMeter:
    """Measure processing FPS without including video opening time."""

    frames: int = 0
    elapsed: float = 0.0

    def tick(self, started_at: float) -> None:
        self.frames += 1
        self.elapsed += perf_counter() - started_at

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed else 0.0
