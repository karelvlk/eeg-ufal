"""Escaped text diffs and an append-only, local review ledger."""
from datetime import datetime, timezone
from difflib import SequenceMatcher
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4
from contextlib import closing

import pandas as pd

LABELS = ["Meaning change / disambiguation", "Added visual information",
          "Equivalent rephrasing", "Possible transcription variation", "Unchanged", "Unclear"]
CONFIDENCES = ["Low", "Medium", "High"]
SNAPSHOT_FIELDS = ["participant", "sentence", "order", "condition", "choice", "source",
                   "initial_translation", "updated_translation", "image_path", "raw_mp3_path"]


def snapshot(row):
    return {key: str(row.get(key, "")) for key in SNAPSHOT_FIELDS}


def trial_id(row):
    return sha256(json.dumps(snapshot(row), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def diff_html(before, after):
    """Return escaped, whitespace-preserving before/after HTML; no meaning inference."""
    left, right = re.findall(r"\s+|\S+", str(before)), re.findall(r"\s+|\S+", str(after))
    outputs = [[], []]
    for operation, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        old, new = escape("".join(left[a:b])), escape("".join(right[c:d]))
        outputs[0].append(old if operation == "equal" else f'<del style="background:#ffe4e6;color:#881337">{old}</del>')
        outputs[1].append(new if operation == "equal" else f'<ins style="background:#dcfce7;color:#14532d;text-decoration:none">{new}</ins>')
    return tuple('<div style="white-space:pre-wrap;line-height:1.8">' + "".join(out) + "</div>" for out in outputs)


def save_review(database, row, reviewer, label, confidence, note):
    if not reviewer.strip() or not note.strip():
        raise ValueError("Reviewer and evidence notes are required.")
    if label not in LABELS or confidence not in CONFIDENCES:
        raise ValueError("Select a review category and confidence.")
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database, timeout=10)) as connection, connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY, trial_id TEXT NOT NULL, reviewer TEXT NOT NULL,
            label TEXT NOT NULL, confidence TEXT NOT NULL, note TEXT NOT NULL,
            created_utc TEXT NOT NULL, snapshot_json TEXT NOT NULL)""")
        connection.execute("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (uuid4().hex, trial_id(row), reviewer.strip(), label, confidence,
                            note.strip(), datetime.now(timezone.utc).isoformat(),
                            json.dumps(snapshot(row), ensure_ascii=False)))


def read_reviews(database):
    columns = ["review_id", "trial_id", "reviewer", "label", "confidence", "note", "created_utc", "snapshot_json"]
    if not Path(database).is_file():
        return pd.DataFrame(columns=columns)
    with closing(sqlite3.connect(database, timeout=10)) as connection:
        exists = connection.execute("SELECT name FROM sqlite_master WHERE name='reviews'").fetchone()
        if not exists:
            return pd.DataFrame(columns=columns)
        return pd.read_sql_query("SELECT * FROM reviews ORDER BY rowid", connection)


def latest_reviews(history):
    return history.drop_duplicates(["trial_id", "reviewer"], keep="last").copy()


def export_table(reviews):
    records = []
    for item in reviews.to_dict("records"):
        stored = json.loads(item.pop("snapshot_json"))
        records.append({**stored, **item, "annotation_status": "Reviewer judgment; not independently validated"})
    return pd.DataFrame(records)


def safe_csv(frame):
    """Neutralize spreadsheet formula strings without changing the raw ledger."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(lambda value: "'" + value if isinstance(value, str)
                                       and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return safe.to_csv(index=False).encode("utf-8-sig")


def evidence_html(row, reviews, gaze_figure=None):
    before, after = diff_html(row.get("initial_translation", ""), row.get("updated_translation", ""))
    notes = "".join(f"<li>{escape(str(r['reviewer']))}: {escape(str(r['label']))} "
                    f"({escape(str(r['confidence']))}) — {escape(str(r['note']))}</li>"
                    for r in reviews.to_dict("records"))
    chart = gaze_figure.to_html(full_html=False, include_plotlyjs=True) if gaze_figure is not None else "<p>No gaze chart available for the selected phase.</p>"
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Translation evidence</title></head><body>"
            "<h1>Translation evidence — reviewer judgments</h1>"
            f"<p>{escape(str(row.get('participant', '')))} / {escape(str(row.get('sentence', '')))} / "
            f"{escape(str(row.get('condition', '')))}</p><p>{escape(str(row.get('source', '')))}</p>"
            f"<p>Assigned image: {escape(str(row.get('image_path', '')))}<br>Evidence ID: {trial_id(row)}</p>"
            f"<h2>Initial</h2>{before}<h2>Final</h2>{after}<h2>Review notes</h2><ul>{notes}</ul>"
            f"<h2>Recording-relative gaze</h2>{chart}"
            "<p>Text differences do not establish semantic correction. No text/image AOIs, "
            "causal effects or synchronized speech events are inferred. "
            "Original images and audio are not embedded. Any gaze chart retains screen coordinates, "
            "not inferred text/image areas.</p></body></html>")
