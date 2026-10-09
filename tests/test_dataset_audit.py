import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_dataset import load_sample
from src.file_loader import FileLoader


class EmptyRecordingTests(unittest.TestCase):
    def test_header_only_signals_are_missing_not_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            eeg = Path(directory) / "eeg.csv"
            gaze = Path(directory) / "gaze.csv"
            eeg.write_text("TimeStamp,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10\n")
            gaze.write_text("TimeStamp,X,Y,Event\n")
            loader = object.__new__(FileLoader)
            with patch("src.dataset_cache.enabled", return_value=False), patch("src.file_loader.st.warning") as warning:
                result = loader.load_data({"eeg": str(eeg), "gaze": str(gaze), "audio": None})
            self.assertEqual(result, (None, None, None))
            self.assertEqual(warning.call_count, 2)
            self.assertIn("no samples", warning.call_args.args[0])

    def test_audit_distinguishes_verified_empty_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaze.csv"
            path.write_text("TimeStamp,X,Y,Event\n")
            with patch("audit_dataset.resolve", return_value=path):
                result = load_sample(directory, "gaze.csv", {"kind": "gaze"})
            self.assertEqual(result["status"], "published_no_samples")
            self.assertEqual(result["rows"], 0)


if __name__ == "__main__":
    unittest.main()
