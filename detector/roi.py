"""Region-of-interest helpers."""

import cv2
import numpy as np

from .contours import DetectedObject


def polygon_array(polygon: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    """Convert a list of points into the OpenCV contour shape."""
    if len(polygon) < 3:
        raise ValueError("ROI polygon needs at least three points")
    return np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))


def create_roi_mask(shape: tuple[int, int], polygon: list[tuple[int, int]]) -> np.ndarray:
    """Create a binary mask for a polygon on an image of (height, width)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_array(polygon)], 255)
    return mask


def apply_roi(mask: np.ndarray, polygon: list[tuple[int, int]] | None) -> np.ndarray:
    """Restrict a binary mask to ROI; an empty ROI leaves the mask unchanged."""
    if not polygon:
        return mask
    return cv2.bitwise_and(mask, create_roi_mask(mask.shape, polygon))


def object_in_roi(obj: DetectedObject, polygon: list[tuple[int, int]] | None) -> bool:
    """Check whether an object's centroid lies inside the ROI polygon."""
    if not polygon:
        return True
    return cv2.pointPolygonTest(polygon_array(polygon), obj.centroid, False) >= 0
