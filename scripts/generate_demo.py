"""Generate a reproducible, annotated demonstration clip without downloads."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate(video: Path, labels: Path, frames: int = 240, fps: int = 25) -> None:
    video.parent.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create demo video: {video}")
    try:
        for index in range(frames):
            frame = np.full((height, width, 3), (35, 45, 55), np.uint8)
            # A mild dynamic background tests stability; it lies outside the
            # configured ROI and therefore must not trigger the detector.
            for reed in range(9):
                x = 10 + reed * 37 + int(2 * np.sin(index / 8 + reed))
                cv2.line(frame, (x, 0), (x + 8, height), (38, 55, 64), 1)
            cv2.rectangle(frame, (0, 170), (width, height), (45, 45, 45), -1)
            x2 = 275 + int(5 * np.sin(index / 5))
            cv2.circle(frame, (x2, 190), 8, (70, 72, 75), -1)
            mask = np.zeros((height, width), np.uint8)
            if index >= 25:
                x = 30 + int((index - 25) * 1.55)
                y = 105 + int(3 * np.sin(index / 9))
                box = (x, y, x + 22, y + 35)
                cv2.rectangle(frame, box[:2], box[2:], (220, 220, 220), -1)
                cv2.circle(frame, (x + 11, y + 8), 5, (245, 245, 245), -1)
                cv2.rectangle(mask, box[:2], box[2:], 255, -1)
                cv2.circle(mask, (x + 11, y + 8), 5, 255, -1)
            cv2.imwrite(str(labels / f"{index:06d}.png"), mask)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--video", type=Path, default=project_root / "data/demo.mp4")
    parser.add_argument("--labels", type=Path, default=project_root / "data/demo_ground_truth")
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args()
    if args.frames < 60:
        parser.error("use at least 60 frames to include background warmup and line crossing")
    generate(args.video, args.labels, args.frames)
    print(f"Created {args.video} and {args.labels} ({args.frames} frames)")


if __name__ == "__main__":
    main()
