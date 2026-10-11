"""Finite stop-line crossings with a dead band and per-track cooldown."""

from dataclasses import asdict, dataclass
from math import hypot

from .tracker import Track


@dataclass(frozen=True)
class CrossingEvent:
    frame: int
    time_seconds: float
    track_id: int
    direction: str
    x: float
    y: float
    kind: str = "line_crossing"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _SideState:
    side: int
    point: tuple[int, int]
    last_event_frame: int | None = None


class LineCrossingCounter:
    """Emit on a stable side change whose path intersects the line segment.

    Signed side = cross(B - A, P - A) / |B - A|, in image coordinates.
    A vertical top-to-bottom line has positive side on its left. The neutral
    band retains the last stable side. New IDs require an initial stable side.
    """

    def __init__(self, line: list[tuple[int, int]], hysteresis: float = 5.0,
                 direction: str = "both", min_hits: int = 3, cooldown_frames: int = 30) -> None:
        self.a, self.b = tuple(line[0]), tuple(line[1])
        self.length = hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])
        if not self.length:
            raise ValueError("line endpoints must differ")
        self.hysteresis = hysteresis
        self.direction = direction
        self.min_hits = min_hits
        self.cooldown_frames = cooldown_frames
        self.states: dict[int, _SideState] = {}
        self.total = 0

    def reset(self) -> None:
        self.states.clear()

    def _signed_distance(self, point: tuple[int, int]) -> float:
        return ((self.b[0] - self.a[0]) * (point[1] - self.a[1])
                - (self.b[1] - self.a[1]) * (point[0] - self.a[0])) / self.length

    def update(self, tracks: list[Track], frame: int, time_seconds: float) -> list[CrossingEvent]:
        live_ids = {track.track_id for track in tracks}
        self.states = {key: value for key, value in self.states.items() if key in live_ids}
        events = []
        for track in tracks:
            if track.missed:
                continue
            distance = self._signed_distance(track.centroid)
            side = 1 if distance > self.hysteresis else -1 if distance < -self.hysteresis else 0
            if not side:
                continue
            state = self.states.get(track.track_id)
            if state is None:
                self.states[track.track_id] = _SideState(side, track.centroid)
                continue
            if side != state.side:
                previous_distance = self._signed_distance(state.point)
                ratio = previous_distance / (previous_distance - distance)
                x = state.point[0] + ratio * (track.centroid[0] - state.point[0])
                y = state.point[1] + ratio * (track.centroid[1] - state.point[1])
                projection = ((x - self.a[0]) * (self.b[0] - self.a[0])
                              + (y - self.a[1]) * (self.b[1] - self.a[1])) / self.length ** 2
                direction = "neg_to_pos" if side > 0 else "pos_to_neg"
                ready = state.last_event_frame is None or frame - state.last_event_frame > self.cooldown_frames
                confirmed = track.streak >= self.min_hits
                if 0 <= projection <= 1 and confirmed and ready and self.direction in ("both", direction):
                    events.append(CrossingEvent(frame, time_seconds, track.track_id, direction, x, y))
                    self.total += 1
                    state.last_event_frame = frame
            # Even an ignored crossing updates the side, avoiding delayed false events.
            state.side, state.point = side, track.centroid
        return events
