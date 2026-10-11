"""Shared processing path used by the CLI and all benchmark methods."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import DetectorConfig
from .alarm import AlarmLogic, AlarmResult
from .background import create_background_subtractor
from .contours import DetectedObject, find_objects
from .line_crossing import CrossingEvent, LineCrossingCounter
from .morphology import clean_mask
from .preprocess import preprocess
from .roi import create_roi_mask, object_in_roi
from .tracker import CentroidTracker, Track


@dataclass
class FrameResult:
    frame_index: int
    time_seconds: float
    blurred: np.ndarray
    raw_mask: np.ndarray
    mask: np.ndarray
    objects: list[DetectedObject]
    tracks: list[Track]
    alarm: AlarmResult
    events: list[CrossingEvent]
    motion_fraction: float
    settling: bool
    illumination_change: bool
    total_crossings: int


class MotionPipeline:
    """Segment -> clean -> filter by ROI -> track -> signal an event."""

    def __init__(self, cfg: DetectorConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.background = create_background_subtractor(cfg)
        self.tracker = CentroidTracker(cfg.tracker_max_distance, cfg.tracker_max_missed, cfg.trajectory_length)
        self.counter = LineCrossingCounter(cfg.line, cfg.line_hysteresis, cfg.direction,
                                          cfg.tracker_min_hits, cfg.cooldown_frames) if cfg.line else None
        self.alarm_logic = AlarmLogic(cfg.min_consecutive_frames, cfg.cooldown_frames)
        self.frame_index = 0
        self.previous: np.ndarray | None = None
        self.roi_mask: np.ndarray | None = None
        self.roi_pixels = 0
        self.recovery_remaining = 0
        self.alarm_remaining = 0

    def _initialize_shape(self, shape: tuple[int, int]) -> None:
        height, width = shape
        for point in self.cfg.roi_polygon + self.cfg.line:
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError(f"ROI/line point {point} outside {width}x{height} frame")
        self.roi_mask = create_roi_mask(shape, self.cfg.roi_polygon) if self.cfg.roi_polygon else np.full(shape, 255, np.uint8)
        self.roi_pixels = int(np.count_nonzero(self.roi_mask))
        if not self.roi_pixels:
            raise ValueError("ROI contains no pixels")

    def process(self, frame: np.ndarray, time_seconds: float = 0.0) -> FrameResult:
        gray = preprocess(frame, self.cfg)
        if self.roi_mask is None:
            self._initialize_shape(gray.shape)
        elif gray.shape != self.roi_mask.shape:
            raise ValueError("video resolution changed during processing")
        illumination_change = False
        if self.previous is not None and self.cfg.illumination_guard:
            delta = gray.astype(np.int16) - self.previous.astype(np.int16)
            selected = delta[self.roi_mask > 0]
            # A majority must brighten or darken together; a moving local object
            # normally changes too few pixels to satisfy this condition.
            bright = np.mean(selected > self.cfg.illumination_delta)
            dark = np.mean(selected < -self.cfg.illumination_delta)
            illumination_change = max(bright, dark) >= self.cfg.illumination_fraction
        if illumination_change:
            self.background = create_background_subtractor(self.cfg)
            self.recovery_remaining = max(1, self.cfg.recovery_frames)
            self.tracker.reset()
            if self.counter:
                self.counter.reset()
            self.alarm_logic = AlarmLogic(self.cfg.min_consecutive_frames, self.cfg.cooldown_frames)
            self.alarm_remaining = 0
        self.previous = gray.copy()
        raw_mask = self.background.apply(gray)
        mask = cv2.bitwise_and(clean_mask(raw_mask, self.cfg), self.roi_mask)
        settling = self.frame_index < self.cfg.warmup_frames or self.recovery_remaining > 0
        if self.recovery_remaining:
            self.recovery_remaining -= 1
        if settling:
            mask = np.zeros_like(mask)
            objects: list[DetectedObject] = []
            tracks: list[Track] = []
        else:
            # Extract complete components first: ROI must not manufacture new
            # centroids at its boundary by clipping a large object in half.
            candidates = find_objects(clean_mask(raw_mask, self.cfg), self.cfg.min_area, self.cfg.max_area)
            objects = [obj for obj in candidates if object_in_roi(obj, self.cfg.roi_polygon)]
            tracks = self.tracker.update(objects)
        events = self.counter.update(tracks, self.frame_index, time_seconds) if self.counter else []
        alarm = self.alarm_logic.update(objects, self.cfg.roi_polygon)
        if self.counter:
            self.alarm_remaining = max(1, self.cfg.min_consecutive_frames) if events else max(0, self.alarm_remaining - 1)
            alarm = AlarmResult(self.alarm_remaining > 0, bool(events), 0, 0, len(objects))
        elif alarm.triggered:
            events.append(CrossingEvent(self.frame_index, time_seconds, -1, "presence", 0, 0, "roi_presence"))
        result = FrameResult(self.frame_index, time_seconds, gray, raw_mask, mask, objects, tracks,
                             alarm, events, float(np.count_nonzero(mask) / self.roi_pixels),
                             settling, illumination_change, self.counter.total if self.counter else 0)
        self.frame_index += 1
        return result
