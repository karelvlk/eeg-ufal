import unittest
import tempfile
from pathlib import Path
import pandas as pd
from src.translation_process import inspect_gaze, image_gaze_points, image_gaze_figure, fixation_heatmap
from PIL import Image
from src.file_loader import FileLoader


class ProcessTests(unittest.TestCase):
    def test_gaze_only_trials_are_indexed_and_load_without_eeg(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "preprocessed-data/gaze/Read"
            target.mkdir(parents=True)
            pd.DataFrame({"X": [100], "Y": [200]}).to_csv(target / "P01_S001_Read.csv", index=False)
            loader = FileLoader(folder)
            self.assertEqual(loader.get_valid_participant_ids("Read"), ["P01"])
            units = loader.get_data_files("Read", "P01", "S001")
            self.assertEqual(len(units), 1)
            self.assertIsNone(units[0]["eeg"])
            eeg, gaze, audio = loader.load_data(units[0])
            self.assertIsNone(eeg)
            self.assertEqual(gaze.X.tolist(), [100])
            self.assertIsNone(audio)

    def test_heatmap_uses_duration_and_preserves_image_orientation(self):
        points = pd.DataFrame({"image_x": [50, 250], "image_y": [50, 150],
                               "duration_seconds": [.1, 1.0]})
        x, y, density = fixation_heatmap(points, (300, 200), smoothing=5)
        self.assertEqual(density.shape, (40, 60))
        self.assertGreater(density[30, 50], density[10, 10] * 5)
        self.assertAlmostEqual(density.max(), 1)
        _, _, empty = fixation_heatmap(pd.DataFrame(), (300, 200))
        self.assertEqual(empty.max(), 0)

    def test_image_mapping_excludes_outside_and_scales_from_screen_rectangle(self):
        runs = pd.DataFrame({"X": [100, 200, 300, 99], "Y": [50, 100, 150, 50],
                             "start_seconds": [0, 1, 2, 3], "duration_seconds": [.1]*4})
        points = image_gaze_points(runs, (100, 50, 200, 100), (400, 200))
        self.assertEqual(points.image_x.tolist(), [0, 200])
        self.assertEqual(points.image_y.tolist(), [0, 100])
        self.assertEqual(runs.X.tolist(), [100, 200, 300, 99])
        figure = image_gaze_figure(Image.new("RGB", (400, 200)), points)
        self.assertEqual(list(figure.layout.yaxis.range), [200, 0])
        self.assertEqual(figure.layout.images[0].yanchor, "top")

    def test_invalid_image_rectangle_is_rejected(self):
        for rectangle in [(0, 0, 0, 100), (0, float("nan"), 100, 100)]:
            with self.assertRaises(ValueError):
                image_gaze_points(pd.DataFrame(), rectangle, (400, 200))

    def test_empty_fraction_is_a_valid_whole_second(self):
        df = pd.DataFrame({"TimeStamp": ["01:00:00.", "01:00:00.001"],
                           "X": [1, 1], "Y": [1, 1], "Event": ["fixation"]*2})
        summary, _ = inspect_gaze(df)
        self.assertEqual(summary["invalid_timestamps"], 0)
        self.assertAlmostEqual(summary["fixation_seconds"], .001)

    def test_fixation_duration_uses_elapsed_time(self):
        df = pd.DataFrame({"TimeStamp": ["01:00:00.000", "01:00:00.010", "01:00:00.020", "01:00:00.030"],
                           "X": [1, 1, 2, 3], "Y": [2, 2, 3, 4],
                           "Event": ["fixation", "fixation", "saccade", "fixation"]})
        summary, runs = inspect_gaze(df)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(summary["fixation_seconds"], .02)

    def test_gap_not_counted_as_fixation(self):
        df = pd.DataFrame({"TimeStamp": ["01:00:00.000", "01:00:00.001", "01:00:01.000", "01:00:01.001"],
                           "X": [1]*4, "Y": [1]*4, "Event": ["fixation"]*4})
        summary, runs = inspect_gaze(df)
        self.assertAlmostEqual(summary["fixation_seconds"], .002)
        self.assertEqual(len(runs), 2)

    def test_invalid_coordinates_and_duplicates(self):
        df = pd.DataFrame({"TimeStamp": ["bad", "01:00:00.000", "01:00:00.000", "01:00:00.001"],
                           "X": [1, ".", 1, 1], "Y": [1]*4, "Event": ["fixation"]*4})
        summary, runs = inspect_gaze(df)
        self.assertEqual(summary["invalid_timestamps"], 1)
        self.assertEqual(summary["duplicate_timestamps"], 1)
        self.assertEqual(summary["fixation_seconds"], 0)

    def test_participant_filter_uses_sentence_argument(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "preprocessed-data/eeg/Read"
            target.mkdir(parents=True)
            (target / "P01_S001_Read.csv").touch()
            (target / "P02_S002_Read.csv").touch()
            loader = FileLoader(folder)
            self.assertEqual(loader.get_valid_participant_ids("Read", "S002"), ["P02"])


if __name__ == "__main__":
    unittest.main()
