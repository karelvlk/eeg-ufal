import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from PIL import Image


class ProcessAppTests(unittest.TestCase):
    def test_signals_can_render_gaze_without_eeg(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "preprocessed-data/gaze/Read"
            target.mkdir(parents=True)
            pd.DataFrame({"TimeStamp": ["01:00:00.000", "01:00:00.010", "01:00:00.020"],
                          "X": [100, 100, 200], "Y": [200, 200, 400],
                          "Event": ["fixation", "fixation", "saccade"]}).to_csv(target / "P01_S001_Read.csv", index=False)
            with patch.dict(os.environ, {"FILE_LOADER_TYPE": "local", "DATA_PATH": str(root), "PROCESS_MANIFEST": ""}):
                import streamlit as st
                st.cache_resource.clear()
                st.cache_data.clear()
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("plotly_chart")), 1)
                self.assertTrue(any("Gaze can be inspected independently" in x.value for x in app.info))
                self.assertFalse(any("No EEG or gaze CSV" in x.value for x in app.warning))

    def test_clean_view_and_verified_overlay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gaze = root / "gaze.csv"
            Image.new("RGB", (300, 500), "white").save(root / "image.jpg")
            pd.DataFrame({"TimeStamp": ["01:00:00.000", "01:00:00.010", "01:00:00.020"],
                          "X": [100, 100, 200], "Y": [200, 200, 400],
                          "Event": ["fixation", "fixation", "saccade"]}).to_csv(gaze, index=False)
            rows = [{"participant": participant, "sentence": "S001", "condition": "A-R",
                     "choice": "changed", "source": "A stand.", "initial_translation": "Old stand.",
                     "updated_translation": "New stand.", "gaze_See_path": "gaze.csv", "image_condition": "R",
                     "image_path": "image.jpg"}
                    for participant in ["P01", "P02"]]
            manifest = root / "trials.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            database = root / "test_reviews.sqlite3"
            settings = {"FILE_LOADER_TYPE": "local", "DATA_PATH": str(root),
                        "PROCESS_MANIFEST": str(manifest), "PROCESS_COMPARISON_MANIFEST": "",
                        "PROCESS_REVIEW_DB": str(database), "NUMBA_CACHE_DIR": str(root / "numba")}
            with patch.dict(os.environ, settings):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.multiselect[0].options), 2)
                self.assertFalse(app.metric)
                self.assertFalse(app.text_input)
                self.assertFalse(any(x.value == "Evidence notebook" for x in app.subheader))
                self.assertFalse(any("ordered fixation" in str(x) for x in app.get("plotly_chart")))
                self.assertEqual(len(app.get("plotly_chart")), 3)
                app.checkbox[0].check().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("plotly_chart")), 3)
                next(x for x in app.selectbox if x.label == "Shared phase").select("Read").run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("plotly_chart")), 2)
                self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
