"""Dependency-free tests: python -m unittest discover -s tests -v"""

import unittest
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from motion_detector.config import DetectorConfig
from motion_detector.detector.alarm import AlarmLogic
from motion_detector.detector.background import FrameDifferenceModel, RunningAverageModel, create_background_subtractor
from motion_detector.detector.contours import find_objects
from motion_detector.detector.line_crossing import LineCrossingCounter
from motion_detector.detector.morphology import clean_mask
from motion_detector.detector.pipeline import MotionPipeline
from motion_detector.detector.preprocess import preprocess
from motion_detector.detector.roi import apply_roi
from motion_detector.detector.tracker import Track
from motion_detector.evaluation.metrics import binary_metrics, event_metrics
from motion_detector.main import run, run_batch
from motion_detector.config import AppConfig
from motion_detector.video_io import find_video_files
from motion_detector.scripts.convert_videos import convert_video


class PipelineTests(unittest.TestCase):
    def test_preprocess_returns_blurred_grayscale(self):
        result = preprocess(np.zeros((20, 30, 3), np.uint8), DetectorConfig())
        self.assertEqual(result.shape, (20, 30))
        self.assertEqual(result.dtype, np.uint8)

    def test_frame_difference_detects_changed_rectangle(self):
        model = FrameDifferenceModel(threshold=10)
        first = np.zeros((50, 50), np.uint8)
        second = first.copy(); second[10:25, 10:25] = 255
        self.assertFalse(model.apply(first).any())
        self.assertGreater(model.apply(second).sum(), 0)

    def test_running_average_detects_motion_before_adapting(self):
        model = RunningAverageModel(alpha=0.01, threshold=10)
        first = np.zeros((30, 30), np.uint8)
        second = first.copy(); second[5:15, 5:15] = 255
        model.apply(first)
        self.assertGreater(model.apply(second).sum(), 0)

    def test_morphology_and_contours_filter_small_noise(self):
        cfg = DetectorConfig(min_area=20, morphology_kernel=3, open_iterations=0, close_iterations=0, dilate_iterations=0)
        mask = np.zeros((60, 60), np.uint8); mask[2, 2] = 255; mask[20:35, 20:35] = 255
        objects = find_objects(clean_mask(mask, cfg), cfg.min_area)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].centroid, (27, 27))

    def test_roi_and_alarm_require_consecutive_frames(self):
        polygon = [(0, 0), (40, 0), (40, 40), (0, 40)]
        mask = np.zeros((50, 50), np.uint8); mask[10:20, 10:20] = 255
        obj = find_objects(apply_roi(mask, polygon), 10)[0]
        alarm = AlarmLogic(min_consecutive_frames=2, cooldown_frames=3)
        self.assertFalse(alarm.update([obj], polygon).triggered)
        self.assertTrue(alarm.update([obj], polygon).triggered)
        self.assertFalse(alarm.update([obj], polygon).triggered)

    def test_line_crossing_emits_once_after_debounced_motion(self):
        counter = LineCrossingCounter([(50, 0), (50, 100)], hysteresis=2, min_hits=2, cooldown_frames=5)
        events = []
        for frame, x in enumerate([40, 42, 48, 52, 58, 60]):
            point = (x, 50)
            events += counter.update([Track(1, point, (x-2, 48, 4, 4), streak=2, history=[point])], frame, frame / 10)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction, "pos_to_neg")
        self.assertAlmostEqual(events[0].x, 50)

    def test_all_background_models_share_factory_api(self):
        frame = np.zeros((20, 20), np.uint8)
        for method in ("mog2", "knn", "diff", "running_average"):
            self.assertEqual(create_background_subtractor(DetectorConfig(method=method)).apply(frame).shape, frame.shape)

    def test_pipeline_warmup_and_line_crossing(self):
        cfg = DetectorConfig(method="diff", min_area=10, warmup_frames=2,
            roi_polygon=[(0, 0), (79, 0), (79, 59), (0, 59)], line=[(40, 5), (40, 55)],
            morphology_kernel=3, open_iterations=0, close_iterations=0, dilate_iterations=0,
            illumination_guard=False, tracker_min_hits=1, tracker_max_distance=30, line_hysteresis=1)
        pipeline = MotionPipeline(cfg)
        blank = np.zeros((60, 80, 3), np.uint8)
        outputs = []
        for index in range(3):
            frame = blank.copy(); x = 20 + index * 10; cv2.rectangle(frame, (x, 20), (x+10, 40), (255, 255, 255), -1)
            outputs.append(pipeline.process(frame, index / 10))
        self.assertTrue(outputs[0].settling and outputs[1].settling)
        self.assertFalse(outputs[2].settling)

    def test_metrics_and_unlabelled_event_metrics(self):
        truth = np.array([[0, 255], [255, 0]], np.uint8)
        scores = binary_metrics(truth, truth)
        self.assertEqual(scores["f1"], 1.0)
        self.assertEqual(scores["iou"], 1.0)
        self.assertNotIn("false_alarms", event_metrics([1.0]))
        self.assertEqual(event_metrics([1.0], [1.1], 3600)["matched_events"], 1.0)


    def test_video_cli_exports_event_statistics_and_video(self):
        base = Path("work") / "motion_detector_test"
        base.mkdir(parents=True, exist_ok=True)
        source = base / "source.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 20, (96, 72))
        self.assertTrue(writer.isOpened())
        for index in range(30):
            frame = np.zeros((72, 96, 3), np.uint8)
            x = 10 + index * 2
            cv2.rectangle(frame, (x, 25), (x+12, 48), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        cfg = DetectorConfig(method="diff", warmup_frames=0, min_area=15,
            roi_polygon=[(0, 0), (95, 0), (95, 71), (0, 71)], line=[(45, 5), (45, 65)],
            line_hysteresis=1, tracker_min_hits=1, tracker_max_distance=20,
            morphology_kernel=3, open_iterations=0, close_iterations=0, dilate_iterations=0,
            illumination_guard=False)
        events, stats, output = base/"events.jsonl", base/"stats.csv", base/"output.avi"
        summary = run(AppConfig(str(source), output, events, stats, headless=True, max_frames=30, output_view="overlay"), cfg)
        self.assertEqual(summary["frames"], 30)
        self.assertTrue(output.exists() and output.stat().st_size > 0)
        self.assertGreaterEqual(len(events.read_text().splitlines()), 1)
        self.assertEqual(len(stats.read_text().splitlines()), 31)
        self.assertEqual(len(list(base.iterdir())), 4)
    def test_global_illumination_change_enters_recovery(self):
        cfg = DetectorConfig(method="mog2", warmup_frames=0, recovery_frames=2, min_area=10,
            roi_polygon=[(0, 0), (39, 0), (39, 29), (0, 29)],
            morphology_kernel=3, open_iterations=0, close_iterations=0, dilate_iterations=0,
            illumination_delta=20, illumination_fraction=0.7)
        pipeline = MotionPipeline(cfg)
        dark = np.zeros((30, 40, 3), np.uint8)
        bright = np.full((30, 40, 3), 180, np.uint8)
        pipeline.process(dark)
        changed = pipeline.process(bright)
        self.assertTrue(changed.illumination_change)
        self.assertTrue(changed.settling)

    def test_invalid_configuration_fails_early(self):
        with self.assertRaises(ValueError):
            DetectorConfig(blur_kernel=4).validate()
        with self.assertRaises(ValueError):
            DetectorConfig(line=[(1, 1)]).validate()

    def test_batch_mode_discovers_and_processes_video_directory(self):
        base = Path("work") / "motion_detector_batch_test"
        base.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            source = base / f"clip_{index}.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(2):
                writer.write(np.zeros((24, 32, 3), np.uint8))
            writer.release()
        self.assertEqual([p.name for p in find_video_files(base)], ["clip_0.avi", "clip_1.avi"])
        result = run_batch(
            base,
            AppConfig(headless=True, max_frames=2),
            DetectorConfig(warmup_frames=0),
            statistics_dir=base / "statistics",
        )
        self.assertEqual(result["videos_found"], 2)
        self.assertEqual(result["videos_processed"], 2)
        self.assertEqual(len(list((base / "statistics").glob("*.csv"))), 2)

    def test_batch_mode_continues_after_unexpected_video_error(self):
        base = Path("work") / "motion_detector_batch_errors"
        base.mkdir(parents=True, exist_ok=True)
        (base / "a_bad.avi").touch()
        (base / "b_good.avi").touch()

        def fake_run(app_cfg, detector_cfg, *, print_summary=True):
            if "a_bad" in app_cfg.video_path:
                raise IndexError("unexpected malformed frame")
            return {"frames": 2, "events": 1}

        with patch("motion_detector.main.run", side_effect=fake_run):
            result = run_batch(base, AppConfig(headless=True), DetectorConfig())

        self.assertEqual(result["videos_processed"], 1)
        self.assertEqual(result["failures"], [{
            "video": str(base / "a_bad.avi"),
            "error": "IndexError: unexpected malformed frame",
        }])

    def test_avi_conversion_preserves_readable_frames(self):
        base = Path("work") / "motion_detector_convert_test"
        base.mkdir(parents=True, exist_ok=True)
        source, converted = base / "source.avi", base / "converted.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        for index in range(4):
            writer.write(np.full((24, 32, 3), index * 40, np.uint8))
        writer.release()

        self.assertEqual(convert_video(source, converted, overwrite=True), 4)
        self.assertTrue(source.exists())
        capture = cv2.VideoCapture(str(converted))
        frame_count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frame_count += 1
        capture.release()
        self.assertEqual(frame_count, 4)


if __name__ == "__main__":
    unittest.main()
