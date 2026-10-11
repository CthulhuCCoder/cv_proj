"""Convert AVI files to MP4 without changing the original dataset."""

import argparse
import json
import os
from pathlib import Path

import cv2

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "motion_detector.scripts"

from ..video_io import find_video_files, resolve_source_path


def convert_video(source: Path, destination: Path, *, overwrite: bool = False) -> int:
    """Convert one readable video and return the number of written frames."""
    if destination.exists() and not overwrite:
        return 0

    temporary = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
    capture = cv2.VideoCapture(str(source))
    writer = None
    frames = 0
    completed = False
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width < 1 or height < 1:
            raise RuntimeError(f"Cannot read video dimensions: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
        writer = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create destination: {temporary}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
        if frames == 0:
            raise RuntimeError(f"Source contains no readable frames: {source}")
        completed = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, destination)
    return frames


def convert_directory(input_dir: Path, output_dir: Path, *, limit: int | None = None,
                      overwrite: bool = False) -> dict:
    """Convert all AVI files recursively, preserving their relative names."""
    files = [path for path in find_video_files(input_dir) if path.suffix.lower() == ".avi"]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No AVI files found in {input_dir}")

    converted = []
    skipped = []
    failures = []
    for source in files:
        destination = output_dir / source.relative_to(input_dir).with_suffix(".mp4")
        if destination.exists() and not overwrite:
            skipped.append(str(source))
            continue
        try:
            frames = convert_video(source, destination, overwrite=overwrite)
            converted.append({"source": str(source), "output": str(destination), "frames": frames})
        except Exception as exc:
            failures.append({"source": str(source), "error": f"{type(exc).__name__}: {exc}"})

    result = {
        "input": str(input_dir),
        "output": str(output_dir),
        "videos_found": len(files),
        "converted": len(converted),
        "skipped": len(skipped),
        "failures": failures,
        "videos": converted,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AVI surveillance clips to MP4")
    parser.add_argument("--input", type=Path, default=Path("videos"), help="AVI directory")
    parser.add_argument("--output", type=Path, default=Path("videos_mp4"), help="MP4 directory")
    parser.add_argument("--limit", type=int, help="convert at most this many files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing MP4 files")
    args = parser.parse_args()
    input_dir = resolve_source_path(str(args.input))
    convert_directory(input_dir, args.output, limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
