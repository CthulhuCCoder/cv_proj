"""One-to-one centroid association with a constant-velocity prediction."""

from dataclasses import dataclass, field
from math import hypot

from .contours import DetectedObject


@dataclass
class Track:
    track_id: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    hits: int = 1
    streak: int = 1
    missed: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)
    history: list[tuple[int, int]] = field(default_factory=list)


class CentroidTracker:
    """Greedy global distance matching; suitable for sparse, unoccluded scenes.

    Only observed tracks may produce crossing events. IDs are not semantic object
    identities and may switch during occlusion, splitting or merging contours.
    """

    def __init__(self, max_distance: float = 80.0, max_missed: int = 8, history_length: int = 30) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.history_length = history_length
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def reset(self) -> None:
        # Keep monotonically increasing IDs across illumination recovery.
        self.tracks.clear()

    def update(self, objects: list[DetectedObject]) -> list[Track]:
        pairs = []
        for track in self.tracks.values():
            gap = track.missed + 1
            predicted = (track.centroid[0] + track.velocity[0] * gap,
                         track.centroid[1] + track.velocity[1] * gap)
            for index, obj in enumerate(objects):
                distance = hypot(predicted[0] - obj.centroid[0], predicted[1] - obj.centroid[1])
                if distance <= self.max_distance:
                    pairs.append((distance, track.track_id, index))
        assigned_tracks: set[int] = set()
        assigned_objects: set[int] = set()
        for _, track_id, index in sorted(pairs):
            if track_id in assigned_tracks or index in assigned_objects:
                continue
            track, obj = self.tracks[track_id], objects[index]
            gap = track.missed + 1
            track.velocity = ((obj.centroid[0] - track.centroid[0]) / gap,
                              (obj.centroid[1] - track.centroid[1]) / gap)
            track.centroid, track.bbox = obj.centroid, obj.bbox
            track.hits += 1
            track.streak = track.streak + 1 if track.missed == 0 else 1
            track.missed = 0
            track.history = (track.history + [obj.centroid])[-self.history_length:]
            assigned_tracks.add(track_id)
            assigned_objects.add(index)
        for track_id in list(self.tracks):
            if track_id not in assigned_tracks:
                track = self.tracks[track_id]
                track.missed += 1
                track.streak = 0
                if track.missed > self.max_missed:
                    del self.tracks[track_id]
        for index, obj in enumerate(objects):
            if index not in assigned_objects:
                self.tracks[self.next_id] = Track(self.next_id, obj.centroid, obj.bbox, history=[obj.centroid])
                self.next_id += 1
        return list(self.tracks.values())
