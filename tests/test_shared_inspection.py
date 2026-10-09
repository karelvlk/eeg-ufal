import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest
from src.process_reviews import trial_id
from src.file_loader import FileLoader
from src.inspection_selection import signal_unit_for_trial


class SharedInspectionTests(unittest.TestCase):
    def test_both_views_preserve_exact_order_condition_and_missing_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "preprocessed-data/gaze/See"
            target.mkdir(parents=True)
            rows = []
            for order, condition in [(1, "A-R"), (2, "A-U")]:
                relative = f"preprocessed-data/gaze/See/P01-{order:02}-S001-{condition}.csv"
                pd.DataFrame({"TimeStamp": ["01:00:00.000", "01:00:00.010", "01:00:00.020"],
                              "X": [600, 610, 620], "Y": [700, 710, 700],
                              "Event": ["fixation"] * 3}).to_csv(root / relative, index=False)
                Image.new("RGB", (480, 360), "white").save(root / f"image{order}.jpg")
                rows.append({"participant": "P01", "sentence": "S001", "order": order,
                             "condition": condition, "choice": "same", "source": "A stand.",
                             "initial_translation": "Old stand.", "updated_translation": "Old stand.",
                             "image_condition": condition[-1], "image_path": f"image{order}.jpg",
                             "gaze_See_path": relative, "gaze_Read_path": ""})
            manifest = root / "trials.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            settings = {"FILE_LOADER_TYPE": "local", "DATA_PATH": str(root),
                        "PROCESS_MANIFEST": str(manifest), "PROCESS_COMPARISON_MANIFEST": "",
                        "NUMBA_CACHE_DIR": str(root / "numba")}
            with patch.dict(os.environ, settings):
                import streamlit as st
                st.cache_resource.clear()
                st.cache_data.clear()
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
                identifier = trial_id(pd.Series(rows[1]))
                app.selectbox(key="inspection_trial").select(identifier).run()
                self.assertFalse(app.exception)
                app.radio[0].set_value("Signals").run()
                self.assertFalse(app.exception)
                self.assertEqual(app.selectbox(key="inspection_trial").value, identifier)
                self.assertTrue(any("P01_S001 · A-U · See · order 2" in c.value for c in app.caption))
                self.assertEqual(next(x for x in app.selectbox if x.label == "Select File").options, ["1. P01-02-S001-A-U"])
                app.selectbox(key="inspection_phase").select("Read").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Read signals are not locally cached" in x.value for x in app.warning))
                self.assertFalse(app.get("plotly_chart"))
                app.radio[0].set_value("Translation process").run()
                self.assertEqual(app.selectbox(key="inspection_trial").value, identifier)
                self.assertEqual(app.selectbox(key="inspection_phase").value, "Read")
                self.assertTrue(any("Read gaze is not cached" in x.value for x in app.warning))
                unit = signal_unit_for_trial(FileLoader(str(root)), pd.Series(rows[1]), "See")
                self.assertEqual(unit["gaze"], str(root / rows[1]["gaze_See_path"]))
                self.assertIsNone(signal_unit_for_trial(FileLoader(str(root)), pd.Series(rows[1]), "Read"))


if __name__ == "__main__":
    unittest.main()
