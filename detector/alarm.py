"""Debounced alarm logic for objects entering an ROI."""

from dataclasses import dataclass

from .contours import DetectedObject
from .roi import object_in_roi


@dataclass(frozen=True)
class AlarmResult:
    """State returned for each processed frame."""

    active: bool
    triggered: bool
    consecutive_frames: int
    cooldown_remaining: int
    objects_in_roi: int


class AlarmLogic:
    """Require persistence in the ROI and limit repeated trigger events."""

    def __init__(self, min_consecutive_frames: int = 3, cooldown_frames: int = 30) -> None:
        if min_consecutive_frames < 1 or cooldown_frames < 0:
            raise ValueError("alarm thresholds must be non-negative and consecutive count must be positive")
        self.min_consecutive_frames = min_consecutive_frames
        self.cooldown_frames = cooldown_frames
        self.consecutive_frames = 0
        self.cooldown_remaining = 0
        self._latched = False

    def update(
        self,
        objects: list[DetectedObject],
        roi_polygon: list[tuple[int, int]] | None = None,
    ) -> AlarmResult:
        in_roi = [obj for obj in objects if object_in_roi(obj, roi_polygon)]
        if in_roi:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0
            self._latched = False

        can_trigger = self.cooldown_remaining == 0
        if self.cooldown_remaining:
            self.cooldown_remaining -= 1
        active = self.consecutive_frames >= self.min_consecutive_frames
        triggered = (
            active and not self._latched and can_trigger
        )
        if triggered:
            self.cooldown_remaining = self.cooldown_frames
            self._latched = True
        return AlarmResult(
            active=active,
            triggered=triggered,
            consecutive_frames=self.consecutive_frames,
            cooldown_remaining=self.cooldown_remaining,
            objects_in_roi=len(in_roi),
        )
