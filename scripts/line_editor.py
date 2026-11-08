"""Interactive OpenCV editor for per-video stop lines."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "motion_detector.scripts"

from ..config import load_line_overrides
from ..main import _line_override_for_video
from ..video_io import find_video_files, resolve_source_path


Point = tuple[int, int]


def line_key(video: Path, directory: Path) -> str:
    """Return a stable, portable key for a video inside ``directory``."""
    try:
        return video.relative_to(directory).as_posix()
    except ValueError:
        return video.name


def read_frame(video: Path, frame_index: int) -> np.ndarray:
    """Read one frame, raising a useful error for malformed sources."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        if frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from: {video}")
        return frame
    finally:
        capture.release()


def save_lines(path: Path, lines: dict[str, list[list[int]] | None]) -> None:
    """Persist the current mapping after every completed video."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(lines.items())), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


@dataclass
class LineEditor:
    """Two-click line editor that converts display coordinates to source pixels."""

    frame: np.ndarray
    title: str
    initial_points: list[Point]
    max_width: int = 1400
    max_height: int = 900

    def __post_init__(self) -> None:
        self.points = list(self.initial_points[:2])
        height, width = self.frame.shape[:2]
        self.scale = min(1.0, self.max_width / width, self.max_height / height)
        self.window = "Line editor"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (round(x / self.scale), round(y / self.scale))
            height, width = self.frame.shape[:2]
            point = (min(max(point[0], 0), width - 1), min(max(point[1], 0), height - 1))
            if len(self.points) == 2:
                self.points = []
            self.points.append(point)
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def _image(self) -> np.ndarray:
        if self.scale < 1.0:
            width = round(self.frame.shape[1] * self.scale)
            height = round(self.frame.shape[0] * self.scale)
            image = cv2.resize(self.frame, (width, height), interpolation=cv2.INTER_AREA)
        else:
            image = self.frame.copy()

        display_points = [(round(x * self.scale), round(y * self.scale)) for x, y in self.points]
        if len(display_points) == 2:
            cv2.arrowedLine(image, display_points[0], display_points[1], (0, 220, 255), 3, tipLength=0.04)
        for index, point in enumerate(display_points, start=1):
            cv2.circle(image, point, 6, (0, 0, 255) if index == 1 else (0, 255, 0), -1)
            cv2.putText(image, str(index), (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        status = "2 points: Enter/S save | N skip | C clear | Q quit"
        if len(self.points) < 2:
            status = f"Click point {len(self.points) + 1} | Right click undo | {status}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 56), (20, 20, 20), -1)
        cv2.putText(image, self.title[:100], (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, status, (8, 43), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 220, 255), 1, cv2.LINE_AA)
        return image

    def run(self) -> str:
        """Return ``save``, ``skip`` or ``quit``."""
        try:
            while True:
                cv2.imshow(self.window, self._image())
                key = cv2.waitKey(30) & 0xFF
                if key in (13, 10, ord("s")):
                    return "save" if len(self.points) == 2 else "empty"
                if key in (ord("n"), ord("k")):
                    return "skip"
                if key in (ord("c"), ord("r")):
                    self.points.clear()
                elif key in (ord("u"), 8):
                    if self.points:
                        self.points.pop()
                elif key in (ord("q"), 27):
                    return "quit"
        finally:
            cv2.destroyWindow(self.window)


def edit_directory(
    directory: Path,
    output: Path,
    *,
    frame_index: int = 0,
    limit: int | None = None,
    start: int = 0,
    files: list[Path] | None = None,
) -> dict:
    """Open each video and save its selected line."""
    all_files = list(files) if files is not None else find_video_files(directory)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if start < 0 or (all_files and start >= len(all_files)):
        raise ValueError("start must point to an existing video index")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    files = all_files[start:]
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No supported video files found in {directory}")

    lines: dict[str, list[list[int]] | None] = {}
    loaded: dict[str, list[tuple[int, int]] | None] = {}
    if output.exists():
        loaded = load_line_overrides(output)
        lines.update({key: [list(point) for point in value] if value else None
                      for key, value in loaded.items()})

    saved = skipped = failures = 0
    try:
        for number, video in enumerate(files, start=start + 1):
            key = line_key(video, directory)
            _, current = _line_override_for_video(video, directory, loaded)
            try:
                frame = read_frame(video, frame_index)
                title = f"[{number}/{len(files) + start}] {video.name}"
                editor = LineEditor(frame, title, current or [])
                action = editor.run()
            except Exception as exc:
                failures += 1
                print(f"[{number}] Failed: {type(exc).__name__}: {exc}")
                continue

            if action in ("save", "empty"):
                lines[key] = [list(point) for point in editor.points] if action == "save" else None
                save_lines(output, lines)
                saved += 1
                print(f"[{number}] Saved {key}: {lines[key]}")
            elif action == "skip":
                skipped += 1
                print(f"[{number}] Skipped {key}")
            else:
                break
    finally:
        cv2.destroyAllWindows()

    result = {"directory": str(directory), "output": str(output), "saved": saved,
              "skipped": skipped, "failures": failures, "lines": len(lines)}
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("videos"), help="video directory or one video")
    parser.add_argument("--output", type=Path, default=Path("configs/video_lines.json"),
                        help="JSON file to create/update")
    parser.add_argument("--frame-index", type=int, default=0, help="frame shown for drawing, default: first")
    parser.add_argument("--limit", type=int, help="edit at most this many videos")
    parser.add_argument("--start", type=int, default=0, help="zero-based video index to start from")
    args = parser.parse_args()

    source = resolve_source_path(str(args.video))
    if source.is_dir():
        edit_directory(source, args.output, frame_index=args.frame_index,
                        limit=args.limit, start=args.start)
    else:
        if args.start or args.limit:
            parser.error("--start and --limit require a video directory")
        edit_directory(source.parent, args.output, frame_index=args.frame_index, files=[source])


if __name__ == "__main__":
    main()
