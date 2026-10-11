"""Interchangeable background models."""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..config import DetectorConfig


class BackgroundModel(ABC):
    """Common interface for all foreground segmentation models."""

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return an 8-bit foreground mask for a grayscale frame."""


class OpenCVBackgroundModel(BackgroundModel):
    def __init__(self, subtractor: cv2.BackgroundSubtractor, learning_rate: float) -> None:
        self._subtractor = subtractor
        self._learning_rate = learning_rate

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return self._subtractor.apply(frame, learningRate=self._learning_rate)


class FrameDifferenceModel(BackgroundModel):
    def __init__(self, threshold: int = 25) -> None:
        self.threshold = threshold
        self.previous: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.previous is None:
            self.previous = frame.copy()
            return np.zeros_like(frame, dtype=np.uint8)
        difference = cv2.absdiff(frame, self.previous)
        self.previous = frame.copy()
        return cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)[1]


class RunningAverageModel(BackgroundModel):
    def __init__(self, alpha: float = 0.02, threshold: int = 25) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.background: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        source = frame.astype(np.float32)
        if self.background is None:
            self.background = source.copy()
            return np.zeros_like(frame, dtype=np.uint8)
        # Classify against the previous background before adapting it.
        background_u8 = cv2.convertScaleAbs(self.background)
        difference = cv2.absdiff(frame, background_u8)
        cv2.accumulateWeighted(source, self.background, self.alpha)
        return cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)[1]


def create_background_subtractor(cfg: DetectorConfig) -> BackgroundModel:
    """Create a configured background model behind a small stable factory API."""
    cfg.validate()
    if cfg.method == "mog2":
        return OpenCVBackgroundModel(
            cv2.createBackgroundSubtractorMOG2(
                history=cfg.history,
                varThreshold=cfg.var_threshold,
                detectShadows=cfg.detect_shadows,
            ),
            cfg.learning_rate,
        )
    if cfg.method == "knn":
        return OpenCVBackgroundModel(
            cv2.createBackgroundSubtractorKNN(
                history=cfg.history,
                dist2Threshold=cfg.knn_dist2_threshold,
                detectShadows=cfg.detect_shadows,
            ),
            cfg.learning_rate,
        )
    if cfg.method == "diff":
        return FrameDifferenceModel(cfg.diff_threshold)
    if cfg.method == "running_average":
        return RunningAverageModel(cfg.average_alpha, cfg.diff_threshold)
    raise ValueError(f"Unknown background method: {cfg.method}")
