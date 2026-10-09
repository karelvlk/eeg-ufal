import unittest
import os
import tempfile
from unittest.mock import patch
import pandas as pd


class GazeReferenceTests(unittest.TestCase):
    def test_reference_handles_singular_samples_without_mutating_data(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch.dict(os.environ, {"NUMBA_CACHE_DIR": cache}):
                from src.processor import create_gaze_kde_heatmap
        frame = pd.DataFrame({"X": [10, 10, "."], "Y": [20, 20, 30],
                              "Event": ["fixation", "fixation", "saccade"]})
        original = frame.copy(deep=True)
        figure = create_gaze_kde_heatmap(frame, image_rectangle=(410, 640, 480, 360))
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(len(figure.data), 0)
        self.assertEqual(list(figure.layout.yaxis2.range), [1080, 0])
        self.assertEqual(figure.layout.shapes[0].x0, 410)
        self.assertEqual(figure.layout.shapes[0].y1, 1000)


if __name__ == "__main__":
    unittest.main()
