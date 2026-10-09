# EMMT EEG and translation-process viewer

Browse UFAL's public [EMMT dataset](https://github.com/ufal/eyetracked-multi-modal-translation)
with two views: **Signals** and **Translation process**. Both use the same exact
trial and phase when prepared metadata is available. Switching views does not
substitute another participant, sentence, condition or presentation order.

## Quick start

Requires Python 3.11+, Poetry and `ffmpeg` on PATH for audio decoding and audits.

```bash
poetry install
poetry run python prepare_full_dataset.py
poetry run streamlit run app.py
```

The bundled `metadata/` contains the previously audited trial inventory and a
complete GitHub file index pinned to commit
`5b685b7a0cb7a3d5a2a0d679e6ee17508106cc5a`. Preparation copies this metadata to
`.cache/full_dataset`; it does **not** download the entire roughly 16 GB corpus.
Files are fetched only when inspected, checked against published Git blob hashes
and cached locally. First access requires internet; verified cached files work
offline. Download failures are not evidence that a recording never existed.

The manifest lists 1,376 rows and 43 participants. The prior audit excludes 147
rows; 19 more eligible rows lack a complete translation pair. These distinctions
are preserved, not replaced by synthetic data. Published phase EEG exists for
27 participants; others can have gaze and audio without usable EEG. An indexed
file is not automatically scientifically usable or quality-controlled.

Some published CSVs contain headers but no samples. These are reported as missing
recording data, not zero EEG or gaze activity.

## Views

**Signals** displays published EEG columns, gaze and available phase audio.
Whole-trial MP3 is separate and is not silently synchronized to EEG. A gaze-only
trial remains browsable when EEG is missing. Per-phase signal plots use independent
relative time axes: shared hover does not prove cross-modal alignment.

**Translation process** displays English source, initial/final Czech transcripts,
escaped text differences, the assigned image, duration-weighted gaze overlay and
whole-trial audio. Compare people on the same source sentence without pooling
different image conditions as matched counterfactuals. Missing files remain visible.

Gaze overlay defaults reconstruct the published layout: image center (650, 820),
native image dimensions, screen 1920 × 1080. This geometry is **not verified against
the recordings**. Coordinates outside the assumed image are excluded. Heatmaps
use approximately 5-pixel bins, 20-pixel Gaussian smoothing and per-trial peak
normalization; red does not measure absolute attention across trials. Read and
Translate do not overlay an image that was not shown in that phase. Signals KDE
weights coordinate samples instead of duration-weighted fixation centres and is
not expected to match the overlay pixel-for-pixel.

Text differences are not validated semantic corrections, spoken decisions,
concentration labels or causal effects. Browsing does not rerun the original
Colab statistics. Existing local review databases are preserved but are not
part of the two-view UI. The experimental Image contribution screen is removed.

## Configuration

Prepared metadata is detected automatically unless explicit environment settings
override it. Remove old `DATA_PATH`/`PROCESS_MANIFEST` settings if they point to a
sparse older dataset. Optional explicit configuration:

```bash
export FILE_LOADER_TYPE=local
export DATA_PATH="$PWD/.cache/full_dataset/data"
export PROCESS_MANIFEST="$PWD/.cache/full_dataset/manifest.csv"
export PROCESS_COMPARISON_MANIFEST="$PROCESS_MANIFEST"
export EMMT_LAZY_DOWNLOAD=1
poetry run streamlit run app.py
```

For a fully offline populated directory, use `FILE_LOADER_TYPE=local`, its root
as `DATA_PATH`, and `EMMT_LAZY_DOWNLOAD=0`. An empty submodule is not a dataset.
For the original remote Signals-only browser, omit the process manifest and set
`FILE_LOADER_TYPE=github` and `DATA_PATH` to
`https://api.github.com/repos/ufal/eyetracked-multi-modal-translation`.

To rebuild with a different audited inventory and matching pinned index:

```bash
poetry run python prepare_full_dataset.py --manifest /path/to/trial_manifest.csv --index /path/to/repository_tree.json
```

## Verification

```bash
poetry run python -m unittest discover -s tests -v
poetry run python audit_dataset.py
# Also fetch representative files across all participants and four phases:
poetry run python audit_dataset.py --fetch-samples
```

The audit checks every linked manifest path and every cached dataset file,
then optionally loads a deterministic EEG/gaze sample covering all participants
with available data, plus representative images, event logs and MP3 files. It
does not claim to have downloaded or inspected every 16 GB file. Its JSON report
is written under `.cache/` and is not committed.

## Before pushing

Commit source, tests, documentation, Poetry files and `metadata/`. Do not commit
`.cache/`, `.venv/`, generated bytecode, secrets or local review databases.
No credentials are required for the public dataset; no audio is sent to a
transcription service by this app. Respect upstream code and dataset licensing;
public availability is not a new licence grant. Research annotations and reports
are maintained separately from this viewer.
