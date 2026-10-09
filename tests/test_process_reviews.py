import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from src.process_reviews import (diff_html, trial_id, save_review, read_reviews,
                                 latest_reviews, safe_csv, evidence_html, export_table)

ROW = {"participant": "P42", "sentence": "S067", "condition": "A-R", "source": "The stand is stable.",
       "initial_translation": "Věšák je stabilní.", "updated_translation": "Stolek je stabilní."}


class ReviewTests(unittest.TestCase):
    def test_diff_is_escaped_and_marks_replacements(self):
        before, after = diff_html("<script>alert(1)</script> old", "<script>alert(1)</script> new")
        self.assertNotIn("<script>", before + after)
        self.assertIn("&lt;script&gt;", before)
        self.assertIn("<del", before)
        self.assertIn("<ins", after)

    def test_unchanged_has_no_markers(self):
        before, after = diff_html("same\ntext", "same\ntext")
        self.assertEqual(before, after)
        self.assertNotIn("<del", before)
        self.assertNotIn("<ins", after)

    def test_row_fingerprint_changes_when_evidence_changes(self):
        self.assertNotEqual(trial_id(ROW), trial_id({**ROW, "updated_translation": "Other"}))
        self.assertNotEqual(trial_id(ROW), trial_id({**ROW, "condition": "A-N"}))

    def test_history_and_independent_reviewers_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reviews.sqlite3"
            self.assertTrue(read_reviews(path).empty)
            save_review(path, ROW, "A", "Unclear", "Low", "Needs audio.")
            save_review(path, ROW, "A", "Meaning change / disambiguation", "Medium", "Two distinct referents.")
            save_review(path, ROW, "B", "Equivalent rephrasing", "Low", "Alternative judgment.")
            history = read_reviews(path)
            current = latest_reviews(history)
            self.assertEqual(len(history), 3)
            self.assertEqual(len(current), 2)
            self.assertEqual(current.trial_id.nunique(), 1)
            self.assertEqual(current[current.reviewer == "A"].iloc[0].label, "Meaning change / disambiguation")
            self.assertEqual(export_table(current).iloc[0].source, ROW["source"])
            self.assertEqual(json.loads(history.iloc[0].snapshot_json)["source"], ROW["source"])

    def test_invalid_review_does_not_create_database(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reviews.sqlite3"
            for reviewer, label, confidence, note in [("", "Unclear", "Low", "note"), ("A", "fake", "Low", "note"), ("A", "Unclear", "Low", "")]:
                with self.assertRaises(ValueError):
                    save_review(path, ROW, reviewer, label, confidence, note)
            self.assertFalse(path.exists())

    def test_csv_formula_safety_and_raw_preservation(self):
        frame = pd.DataFrame({"note": ["=BAD()", "normal", " @BAD()"]})
        csv = safe_csv(frame).decode("utf-8-sig")
        self.assertIn("'=BAD()", csv)
        self.assertIn("' @BAD()", csv)
        self.assertEqual(frame.note.iloc[0], "=BAD()")

    def test_html_card_escapes_notes(self):
        reviews = pd.DataFrame([{"reviewer": "A", "label": "Unclear", "confidence": "Low", "note": "<img src=x onerror=evil()>"}])
        html = evidence_html(ROW, reviews)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)
        self.assertIn("Věšák", html)
        self.assertIn("Stolek", html)

    def test_evidence_card_embeds_gaze_chart(self):
        import plotly.graph_objects as go
        figure = go.Figure(go.Scatter(x=[0, 1], y=[200, 500]))
        html = evidence_html(ROW, pd.DataFrame(), figure)
        self.assertIn("Plotly.newPlot", html)
        self.assertIn(trial_id(ROW), html)


if __name__ == "__main__":
    unittest.main()
