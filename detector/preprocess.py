"""Frame preprocessing."""

import cv2
import numpy as np

from ..config import DetectorConfig


def preprocess(frame: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """Convert BGR frame to blurred grayscale image."""
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty image")
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0)
