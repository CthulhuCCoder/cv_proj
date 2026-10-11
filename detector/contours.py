"""Object extraction from a binary foreground mask."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedObject:
    """One connected foreground component after area filtering."""

    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    centroid: tuple[int, int]


def find_objects(mask: np.ndarray, min_area: float, max_area: float | None = None) -> list[DetectedObject]:
    """Find external contours and keep components within the configured area range."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects: list[DetectedObject] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or (max_area is not None and area > max_area):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"]:
            centroid = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
        else:
            centroid = (x + width // 2, y + height // 2)
        objects.append(DetectedObject(contour, (x, y, width, height), area, centroid))
    return sorted(objects, key=lambda item: item.area, reverse=True)
