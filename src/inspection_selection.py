"""One exact manifest trial/phase shared by both inspection views."""
from pathlib import Path
import pandas as pd
from src.process_reviews import trial_id
from src.dataset_cache import available as dataset_available, resolve_ui


def select_inspection_trial(manifest_path, root):
    import streamlit as st
    if not manifest_path or not Path(manifest_path).is_file():
        return None
    try:
        trials = pd.read_csv(manifest_path, keep_default_na=False)
        required = {"participant", "sentence", "condition", "choice", "source",
                    "initial_translation", "updated_translation", "gaze_See_path", "image_condition"}
        if not required <= set(trials):
            st.warning("The process manifest cannot provide a shared trial selection: required columns are missing.")
            return None
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"Shared trial selection could not be loaded: {exc}")
        return None
    available = trials.copy()
    if available.empty:
        st.warning("No local See gaze files match the shared trial manifest.")
        return None
    available["identity"] = available.apply(trial_id, axis=1)
    available = available.drop_duplicates("identity")
    condition = st.sidebar.selectbox("Image condition", ["All"] + sorted(available.image_condition.unique()), key="inspection_condition")
    if condition != "All":
        available = available[available.image_condition.eq(condition)]
    choices = available.identity.tolist()
    if st.session_state.get("inspection_trial") not in choices:
        st.session_state.inspection_trial = choices[0]
    indexed = available.set_index("identity", drop=False)
    def label(identifier):
        row = indexed.loc[identifier]
        return f"{row.participant}_{row.sentence} · {row.condition} · order {row.get('order', '')}"
    identifier = st.sidebar.selectbox("Shared trial", choices, format_func=label, key="inspection_trial")
    phase = st.sidebar.selectbox("Shared phase", ["See", "Read", "Translate", "Update"], key="inspection_phase")
    st.sidebar.caption("Both views use this exact trial and phase. Missing recordings never select a different trial.")
    row = indexed.loc[identifier]
    st.sidebar.caption(f"Manifest: {len(trials)} trials, {trials.participant.nunique()} participants.")
    st.sidebar.caption("Files for this phase: " + ", ".join(f"{kind}: {'available' if dataset_available(root, str(row.get(f'{kind}_{phase}_path', ''))) else 'missing'}" for kind in ('eeg', 'gaze')))
    return indexed.loc[identifier], phase


def signal_unit_for_trial(loader, row, phase):
    """Resolve exact manifest paths, with index fallback only for the same basename."""
    root = Path(loader.base_path)
    gaze_path = str(row.get(f"gaze_{phase}_path", ""))
    reference = gaze_path or str(row.get("gaze_See_path", ""))
    basename = Path(reference).stem
    def local(path):
        resolved = resolve_ui(root, path) if path else None
        return str(resolved) if resolved else None
    eeg_path = str(row.get(f"eeg_{phase}_path", ""))
    if not eeg_path:
        eeg_path = loader.eeg_files[phase].get(basename, {}).get("path", "")
    audio_path = loader.audio_files[phase].get(basename, "")
    unit = {"name": basename, "participant_id": row.participant, "sentence_id": row.sentence,
            "category": phase, "eeg": local(eeg_path), "gaze": local(gaze_path), "audio": local(audio_path),
            "eeg_source_path": eeg_path}
    return unit if unit["eeg"] or unit["gaze"] else None
