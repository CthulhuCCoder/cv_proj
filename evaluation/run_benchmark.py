"""Compare background methods using the production MotionPipeline."""

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

# Support both `python -m motion_detector.evaluation.run_benchmark` and direct
# execution from a checkout, as shown in the project README.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "motion_detector.evaluation"

from ..config import DetectorConfig
from ..detector.pipeline import MotionPipeline
from .metrics import binary_metrics, event_metrics


def _find_truth(directory: Path, index: int) -> Path | None:
    for candidate in (directory / f"{index:06d}.png", directory / f"frame_{index:06d}.png", directory / f"{index}.png"):
        if candidate.exists():
            return candidate
    return None


def run_method(video: str, method: str, template: DetectorConfig, ground_truth_dir: Path | None,
               max_frames: int | None, expected_event_times: list[float] | None = None) -> dict[str, float | str]:
    capture = cv2.VideoCapture(0 if video == "0" else video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cfg = DetectorConfig(**{name: getattr(template, name) for name in template.__dataclass_fields__})
    cfg.method = method
    pipeline = MotionPipeline(cfg)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    scores: list[dict[str, float]] = []
    pixel_counts = {"tp": 0, "fp": 0, "fn": 0}
    event_times: list[float] = []
    frame_truth: list[bool] = []
    frame_predictions: list[bool] = []
    index = 0
    processing = 0.0
    while max_frames is None or index < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        started = perf_counter()
        result = pipeline.process(frame, index / fps)
        processing += perf_counter() - started
        event_times.extend(event.time_seconds for event in result.events)
        if ground_truth_dir:
            truth_path = _find_truth(ground_truth_dir, index)
            if truth_path:
                truth = cv2.imread(str(truth_path), cv2.IMREAD_GRAYSCALE)
                if truth is not None:
                    scores.append(binary_metrics(result.mask, truth))
                    pred_fg, truth_fg = result.mask > 0, truth > 0
                    pixel_counts["tp"] += int(np.count_nonzero(pred_fg & truth_fg))
                    pixel_counts["fp"] += int(np.count_nonzero(pred_fg & ~truth_fg))
                    pixel_counts["fn"] += int(np.count_nonzero(~pred_fg & truth_fg))
                    frame_truth.append(bool(np.any(truth)))
                    frame_predictions.append(bool(result.objects))
        index += 1
    capture.release()
    result: dict[str, float | str] = {"method": method, "frames": float(index), "fps": index / processing if processing else 0.0}
    if scores:
        tp, fp, fn = pixel_counts["tp"], pixel_counts["fp"], pixel_counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        result.update({"pixel_precision": precision, "pixel_recall": recall,
                       "pixel_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                       "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                       "mean_frame_iou": float(np.mean([score["iou"] for score in scores]))})
    if frame_truth:
        tp = sum(actual and predicted for actual, predicted in zip(frame_truth, frame_predictions))
        fp = sum((not actual) and predicted for actual, predicted in zip(frame_truth, frame_predictions))
        fn = sum(actual and (not predicted) for actual, predicted in zip(frame_truth, frame_predictions))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        result.update({"frame_precision": precision, "frame_recall": recall,
                       "frame_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    result.update(event_metrics(event_times, expected_times=expected_event_times, duration_seconds=index / fps))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--event-times", type=float, nargs="*",
                        help="ground-truth crossing times in seconds; enables false-alarm and delay metrics")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    template = DetectorConfig.from_json(args.config) if args.config else DetectorConfig()
    rows = [run_method(args.video, method, template, args.ground_truth_dir, args.max_frames, args.event_times)
            for method in ("mog2", "knn", "diff", "running_average")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
