"""Command-line video processing, event JSONL and per-frame CSV export."""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

# Support both `python main.py` and `python -m motion_detector`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "motion_detector"

import cv2

from .config import AppConfig, DetectorConfig, METHODS
from .detector.pipeline import MotionPipeline
from .visualization import draw_overlay, make_diagnostic_view, save_snapshot
from .video_io import find_video_files, resolve_source_path


def _parse_points(values: list[str] | None) -> list[tuple[int, int]]:
    points = []
    for value in values or []:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"Point must use x,y notation: {value}")
        points.append((int(parts[0]), int(parts[1])))
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical motion and stop-line crossing detector (CPU)")
    parser.add_argument("--video", default="0", help="video path, directory, or camera index")
    parser.add_argument("--config", type=Path, help="JSON configuration; CLI values override it")
    parser.add_argument("--save-config", type=Path, help="write resolved algorithm parameters")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--output", type=Path, help="annotated .mp4 or .avi video")
    parser.add_argument("--output-view", choices=("diagnostic", "overlay"), default="diagnostic")
    parser.add_argument("--events", type=Path, help="event log in JSONL format")
    parser.add_argument("--statistics", type=Path, help="per-frame statistics CSV")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("results/snapshots"))
    parser.add_argument("--snapshot-every", type=int, default=0, help="save every N frames; 0 disables")
    parser.add_argument("--roi", nargs="+", help="ROI vertices: x,y x,y x,y ...")
    parser.add_argument("--line", nargs=2, help="finite stop-line endpoints: x1,y1 x2,y2")
    parser.add_argument("--direction", choices=("both", "neg_to_pos", "pos_to_neg"))
    parser.add_argument("--min-area", type=float)
    parser.add_argument("--min-consecutive", type=int, dest="min_consecutive_frames")
    parser.add_argument("--cooldown", type=int, dest="cooldown_frames")
    parser.add_argument("--warmup", type=int, dest="warmup_frames")
    parser.add_argument("--history", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--blur-kernel", type=int)
    parser.add_argument("--no-illumination-guard", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--playback", action="store_true", help="pace GUI playback at source FPS")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-dir", type=Path, help="directory for annotated videos in batch mode")
    parser.add_argument("--events-dir", type=Path, help="directory for JSONL events in batch mode")
    parser.add_argument("--statistics-dir", type=Path, help="directory for CSV statistics in batch mode")
    parser.add_argument("--limit-videos", type=int, help="process at most this many files in batch mode")
    return parser


def run(app_cfg: AppConfig, detector_cfg: DetectorConfig, *, print_summary: bool = True) -> dict:
    """Process a source once; release all video and log resources on failures."""
    detector_cfg.validate()
    if app_cfg.max_frames is not None and app_cfg.max_frames < 1:
        raise ValueError("max_frames must be positive")
    if app_cfg.snapshot_every < 0:
        raise ValueError("snapshot_every must be non-negative")
    if app_cfg.output_path and app_cfg.output_path.suffix.lower() not in (".avi", ".mp4"):
        raise ValueError("output extension must be .avi or .mp4")
    destinations = [p.resolve() for p in (app_cfg.output_path, app_cfg.events_path, app_cfg.statistics_path) if p]
    if len(set(destinations)) != len(destinations):
        raise ValueError("video, event and statistics outputs must use different paths")
    if not app_cfg.video_path.isdecimal() and Path(app_cfg.video_path).resolve() in destinations:
        raise ValueError("output must not overwrite the input video")
    source = int(app_cfg.video_path) if app_cfg.video_path.isdecimal() else app_cfg.video_path
    capture = cv2.VideoCapture(source)
    window_name = f"Motion detector - {Path(app_cfg.video_path).name}"
    writer = event_handle = stats_handle = None
    processed = events_count = 0
    processing_seconds = 0.0
    started_all = perf_counter()
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {app_cfg.video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        pipeline = MotionPipeline(detector_cfg)
        if app_cfg.events_path:
            app_cfg.events_path.parent.mkdir(parents=True, exist_ok=True)
            event_handle = app_cfg.events_path.open("w", encoding="utf-8")
        stats_writer = None
        if app_cfg.statistics_path:
            app_cfg.statistics_path.parent.mkdir(parents=True, exist_ok=True)
            stats_handle = app_cfg.statistics_path.open("w", newline="", encoding="utf-8")
            fields = ["frame", "time_seconds", "motion_fraction", "objects", "alarm", "events", "crossings", "settling", "illumination_change", "processing_ms"]
            stats_writer = csv.DictWriter(stats_handle, fieldnames=fields)
            stats_writer.writeheader()
        while app_cfg.max_frames is None or processed < app_cfg.max_frames:
            loop_started = perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            # File timestamps are derived from zero-based index / source FPS.
            timestamp = perf_counter() - started_all if isinstance(source, int) else processed / fps
            started = perf_counter()
            result = pipeline.process(frame, timestamp)
            duration = perf_counter() - started
            processing_seconds += duration
            events_count += len(result.events)
            if event_handle:
                for event in result.events:
                    event_handle.write(json.dumps(event.to_dict()) + "\n")
                if result.events:
                    event_handle.flush()
            if stats_writer:
                stats_writer.writerow(dict(frame=processed, time_seconds=f"{timestamp:.6f}",
                                           motion_fraction=f"{result.motion_fraction:.6f}", objects=len(result.objects),
                                           alarm=int(result.alarm.active), events=len(result.events), crossings=result.total_crossings,
                                           settling=int(result.settling), illumination_change=int(result.illumination_change),
                                           processing_ms=f"{duration * 1000:.4f}"))
            snapshot_due = app_cfg.snapshot_every and processed % app_cfg.snapshot_every == 0
            if app_cfg.output_path or not app_cfg.headless or snapshot_due:
                view = make_diagnostic_view(frame, result, detector_cfg) if app_cfg.output_view == "diagnostic" else draw_overlay(frame, result, detector_cfg)
                if app_cfg.output_path:
                    if writer is None:
                        app_cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
                        codec = "MJPG" if app_cfg.output_path.suffix.lower() == ".avi" else "mp4v"
                        writer = cv2.VideoWriter(str(app_cfg.output_path), cv2.VideoWriter_fourcc(*codec), fps, (view.shape[1], view.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot open video writer: {app_cfg.output_path} ({codec})")
                    writer.write(view)
                if snapshot_due:
                    save_snapshot(view, app_cfg.snapshot_dir, processed)
                if not app_cfg.headless:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, view)
                    delay = max(1, round(1000 / fps - (perf_counter() - loop_started) * 1000)) if app_cfg.playback else 1
                    key = cv2.waitKey(delay) & 0xFF
                    if key == ord("s"):
                        save_snapshot(view, app_cfg.snapshot_dir, processed)
                    if key in (ord("q"), 27):
                        processed += 1
                        break
            processed += 1
        if processed == 0:
            raise RuntimeError("Source opened but contains no readable frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if event_handle is not None:
            event_handle.close()
        if stats_handle is not None:
            stats_handle.close()
        if not app_cfg.headless:
            cv2.destroyAllWindows()
    elapsed = perf_counter() - started_all
    summary = dict(method=detector_cfg.method, frames=processed, events=events_count,
                   processing_fps=processed / processing_seconds if processing_seconds else 0,
                   end_to_end_fps=processed / elapsed if elapsed else 0,
                   processing_ms=1000 * processing_seconds / processed)
    if print_summary:
        print(json.dumps(summary, indent=2))
    return summary


def run_batch(
    directory: Path,
    app_cfg: AppConfig,
    detector_cfg: DetectorConfig,
    *,
    output_dir: Path | None = None,
    events_dir: Path | None = None,
    statistics_dir: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Process every supported video in ``directory`` and return an aggregate."""
    if limit is not None and limit < 1:
        raise ValueError("limit_videos must be positive")
    files = find_video_files(directory)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No supported video files found in {directory}")

    summaries = []
    failures = []
    for index, video in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Processing: {video.name}", flush=True)
        stem = video.stem
        per_video = replace(
            app_cfg,
            video_path=str(video),
            output_path=output_dir / f"{stem}.avi" if output_dir else None,
            events_path=events_dir / f"{stem}.jsonl" if events_dir else None,
            statistics_path=statistics_dir / f"{stem}.csv" if statistics_dir else None,
            snapshot_dir=app_cfg.snapshot_dir / stem if app_cfg.snapshot_every else app_cfg.snapshot_dir,
        )
        try:
            summary = run(per_video, detector_cfg, print_summary=False)
            summaries.append({"video": str(video), **summary})
            print(f"[{index}/{len(files)}] Done: {summary['frames']} frames", flush=True)
        except Exception as exc:
            # A single malformed AVI must not stop a long dataset run. The
            # exact exception type is kept in the aggregate for diagnosis.
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"video": str(video), "error": error})
            print(f"[{index}/{len(files)}] Failed: {error}", flush=True)

    result = {
        "directory": str(directory),
        "videos_found": len(files),
        "videos_processed": len(summaries),
        "frames": sum(item["frames"] for item in summaries),
        "events": sum(item["events"] for item in summaries),
        "failures": failures,
        "videos": summaries,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        cfg = DetectorConfig.from_json(args.config) if args.config else DetectorConfig()
        for name in asdict(cfg):
            value = getattr(args, name, None)
            if value is not None and name != "line":
                setattr(cfg, name, value)
        if args.roi is not None:
            cfg.roi_polygon = _parse_points(args.roi)
        if args.line is not None:
            cfg.line = _parse_points(args.line)
        if args.no_illumination_guard:
            cfg.illumination_guard = False
        cfg.validate()
        if args.save_config:
            cfg.to_json(args.save_config)
        source = resolve_source_path(args.video)
        if source.is_dir() and any((args.output, args.events, args.statistics)):
            raise ValueError("batch mode uses --output-dir, --events-dir and --statistics-dir")
        app = AppConfig(str(source), args.output, args.events, args.statistics, args.snapshot_dir,
                        args.snapshot_every, args.headless, args.max_frames, args.output_view, args.playback)
        if source.is_dir():
            run_batch(source, app, cfg, output_dir=args.output_dir, events_dir=args.events_dir,
                      statistics_dir=args.statistics_dir, limit=args.limit_videos)
        else:
            run(app, cfg)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
