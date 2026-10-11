"""Foreground mask cleanup."""

import cv2
import numpy as np

from ..config import DetectorConfig


def clean_mask(mask: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """Remove shadows and sensor noise, then close holes and dilate regions."""
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")
    binary = np.where(mask >= cfg.shadow_threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morphology_kernel, cfg.morphology_kernel)
    )
    if cfg.open_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=cfg.open_iterations)
    if cfg.close_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iterations)
    if cfg.dilate_iterations:
        binary = cv2.dilate(binary, kernel, iterations=cfg.dilate_iterations)
    return binary
