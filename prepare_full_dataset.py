"""Prepare full-corpus metadata from the existing audited EMMT manifest.

Does not download the 16 GB corpus and does not calculate research results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

def prepare(manifest, tree_path, output):
    payload = json.loads(Path(tree_path).read_text())
    if payload.get("truncated") or len(payload.get("sha", "")) != 40:
        raise ValueError("Incomplete repository tree")
    files = {i["path"]: i for i in payload["tree"] if i.get("type") == "blob"}
    with Path(manifest).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    required = {"participant", "sentence", "order", "condition", "raw_mp3_path", "gaze_See_path"}
    if not required <= set(fields):
        raise ValueError("An audited trial manifest is required")
    for field in ("image_path", "dataset_commit"):
        if field not in fields:
            fields.append(field)
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ("participant", "sentence", "order", "condition"))
        if key in seen:
            raise ValueError(f"Duplicate trial identity: {key}")
        seen.add(key)
        for key, value in row.items():
            if key.endswith("_path") and value and value not in files:
                raise ValueError(f"Manifest path not in pinned tree: {value}")
        match = re.search(r"-i([^/.]+)\.mp3$", row["raw_mp3_path"])
        image = f"images/{match.group(1)}.jpg" if match else ""
        row["image_path"] = image if image in files else ""
        row["dataset_commit"] = payload["sha"]
    output = Path(output)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "data/repository_tree.json").write_text(json.dumps(payload))
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"dataset_commit": payload["sha"], "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                  "trials": len(rows), "participants": len({r["participant"] for r in rows}),
                  "images_linked": sum(bool(r["image_path"]) for r in rows),
                  "warning": "Metadata availability is not quality-control eligibility. Image geometry remains unverified."}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parent
    parser.add_argument("--manifest", default=str(project / "metadata/trial_manifest.csv"))
    parser.add_argument("--index", default=str(project / "metadata/repository_tree.json"))
    parser.add_argument("--output", default=str(project / ".cache/full_dataset"))
    args = parser.parse_args()
    prepare(args.manifest, args.index, args.output)
