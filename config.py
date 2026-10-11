"""Validated configuration, optionally loaded from a JSON file."""

import json
from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from pathlib import Path
from typing import Literal

Method = Literal["mog2", "knn", "diff", "running_average"]
METHODS = ("mog2", "knn", "diff", "running_average")


@dataclass
class DetectorConfig:
    """All algorithm parameters; coordinates refer to the source resolution."""

    method: Method = "mog2"
    blur_kernel: int = 5
    shadow_threshold: int = 200
    history: int = 500
    var_threshold: float = 16.0
    knn_dist2_threshold: float = 400.0
    detect_shadows: bool = True
    learning_rate: float = -1.0
    diff_threshold: int = 25
    average_alpha: float = 0.02
    morphology_kernel: int = 5
    open_iterations: int = 1
    close_iterations: int = 2
    dilate_iterations: int = 1
    min_area: float = 500.0
    max_area: float | None = None
    min_consecutive_frames: int = 3
    cooldown_frames: int = 30
    warmup_frames: int = 30
    illumination_guard: bool = True
    illumination_delta: float = 25.0
    illumination_fraction: float = 0.65
    recovery_frames: int = 15
    roi_polygon: list[tuple[int, int]] = field(default_factory=list)
    line: list[tuple[int, int]] = field(default_factory=list)
    direction: str = "both"
    line_hysteresis: float = 5.0
    tracker_max_distance: float = 80.0
    tracker_max_missed: int = 8
    tracker_min_hits: int = 3
    trajectory_length: int = 30

    def validate(self) -> None:
        """Reject invalid configuration before allocating OpenCV resources."""
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        for name in ("blur_kernel", "morphology_kernel"):
            value = getattr(self, name)
            if type(value) is not int or value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        for name in ("history", "min_consecutive_frames", "tracker_min_hits", "trajectory_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("open_iterations", "close_iterations", "dilate_iterations", "cooldown_frames",
                     "warmup_frames", "recovery_frames", "tracker_max_missed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("min_area", "var_threshold", "knn_dist2_threshold", "line_hysteresis",
                     "tracker_max_distance", "illumination_delta"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.var_threshold == 0 or self.knn_dist2_threshold == 0 or self.tracker_max_distance == 0:
            raise ValueError("background thresholds and tracker_max_distance must be positive")
        if self.max_area is not None and (not isfinite(self.max_area) or self.max_area < self.min_area):
            raise ValueError("max_area must be finite and >= min_area")
        for name, lower, upper in (("shadow_threshold", 1, 255), ("diff_threshold", 0, 254)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if not (self.learning_rate == -1 or 0 <= self.learning_rate <= 1):
            raise ValueError("learning_rate must be -1 (automatic) or in [0, 1]")
        if not 0 < self.average_alpha <= 1:
            raise ValueError("average_alpha must be in (0, 1]")
        if not 0 < self.illumination_fraction <= 1:
            raise ValueError("illumination_fraction must be in (0, 1]")
        for name in ("detect_shadows", "illumination_guard"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        if self.direction not in ("both", "neg_to_pos", "pos_to_neg"):
            raise ValueError("direction must be both, neg_to_pos or pos_to_neg")
        for name, allowed in (("roi_polygon", None), ("line", 2)):
            points = getattr(self, name)
            if not isinstance(points, (list, tuple)):
                raise ValueError(f"{name} must be a list of [x, y] points")
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2 or any(type(v) is not int or v < 0 for v in point):
                    raise ValueError(f"{name} coordinates must be non-negative integers")
            if points and allowed and len(points) != allowed:
                raise ValueError("line needs exactly two endpoints")
        if self.line and tuple(self.line[0]) == tuple(self.line[1]):
            raise ValueError("line endpoints must be different")
        if self.roi_polygon:
            p = self.roi_polygon
            area2 = sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))
            if len(p) < 3 or area2 == 0:
                raise ValueError("ROI needs at least three vertices with nonzero polygon area")

    @classmethod
    def from_json(cls, path: Path) -> "DetectorConfig":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = data.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config = cls(**data)
        config.validate()
        return config

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class AppConfig:
    """I/O and display settings for the command-line application."""

    video_path: str = "0"
    output_path: Path | None = None
    events_path: Path | None = None
    statistics_path: Path | None = None
    snapshot_dir: Path = Path("results/snapshots")
    snapshot_every: int = 0
    headless: bool = False
    max_frames: int | None = None
    output_view: str = "diagnostic"
    playback: bool = False

