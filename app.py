import os
import streamlit as st
import src.processor as processor
import pandas as pd
from src.file_loader import GithubFileLoader, FileLoader
import plotly.graph_objects as go
from pathlib import Path
from src.dataset_cache import resolve_ui

# Prepared metadata lists the whole corpus; large files are cached on selection.
prepared = Path(__file__).resolve().parent / ".cache/full_dataset"
if (prepared / "manifest.csv").is_file() and (prepared / "data/repository_tree.json").is_file():
    os.environ.setdefault("FILE_LOADER_TYPE", "local")
    os.environ.setdefault("DATA_PATH", str(prepared / "data"))
    os.environ.setdefault("PROCESS_MANIFEST", str(prepared / "manifest.csv"))
    os.environ.setdefault("PROCESS_COMPARISON_MANIFEST", str(prepared / "manifest.csv"))
    if Path(os.environ["DATA_PATH"]).resolve() == (prepared / "data").resolve():
        os.environ.setdefault("EMMT_LAZY_DOWNLOAD", "1")

# Set page configuration
st.set_page_config(page_title="EEG Data Browser", layout="wide")

st.session_state.pop("contribution_next_view", None)
if st.session_state.get("main_view") not in (None, "Signals", "Translation process"):
    st.session_state.pop("main_view")
view = st.sidebar.radio("View", ["Signals", "Translation process"],
                        index=int(bool(os.getenv("PROCESS_MANIFEST"))), key="main_view")
from src.inspection_selection import select_inspection_trial, signal_unit_for_trial
shared_selection = None
if os.getenv("FILE_LOADER_TYPE", "github") == "local":
    shared_selection = select_inspection_trial(os.getenv("PROCESS_MANIFEST", ""), Path(os.getenv("DATA_PATH", "ufal_emmt")))
if view == "Translation process":
    from src.translation_process import render
    render(shared_selection)
    st.stop()


# Singleton implementation for loader
@st.cache_resource
def get_loader():
    file_loader_type = os.getenv("FILE_LOADER_TYPE", "github")
    data_path = os.getenv("DATA_PATH") or (
        str(Path(__file__).resolve().parent / "ufal_emmt") if file_loader_type == "local"
        else "https://api.github.com/repos/ufal/eyetracked-multi-modal-translation"
    )
    print(f"file_loader_type: {file_loader_type}")
    print(f"data_path: {data_path}")
    if file_loader_type == "local":
        return FileLoader(base_path=data_path)
    elif file_loader_type == "github":
        return GithubFileLoader(base_url=data_path)
    else:
        raise ValueError(f"Invalid file loader type: {file_loader_type}")


# Get the singleton instance
try:
    loader = get_loader()
except (RuntimeError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.info("You can still open Translation process from the sidebar. For offline Signals, set FILE_LOADER_TYPE=local and DATA_PATH to your populated dataset folder.")
    if st.button("Retry loading dataset"):
        get_loader.clear()
        st.rerun()
    st.stop()
if getattr(loader, "index_warning", None):
    st.warning(loader.index_warning)
if st.sidebar.button("Refresh dataset index"):
    get_loader.clear()
    st.cache_data.clear()
    st.rerun()


@st.cache_data
def create_files_dataframe(data_units: list):
    data = []
    for du in data_units:
        data.append(
            {
                "Filename": du["name"],
                "Category": du["category"],
                "Participant": du["participant_id"],
                "Sentence": du["sentence_id"],
            }
        )
    return pd.DataFrame(data)


@st.cache_data
def get_participant_ids(category, sentence_filter):
    return ["All"] + loader.get_valid_participant_ids(
        category,
        sentence_id_filter=(sentence_filter if sentence_filter != "All" else None),
    )


@st.cache_data
def get_sentence_ids(category, participant_filter):
    return ["All"] + loader.get_valid_sentence_ids(
        category,
        participant_id_filter=(participant_filter if participant_filter != "All" else None),
    )


@st.cache_data
def load_file_data(data_unit):
    return loader.load_data(data_unit)


@st.cache_data
def process_and_plot_data_cached(
    file_name,
    eeg_df,
    audio_io,
    gaze_df,
    window_size,
    column_range,
    ica=None,
    bandpass=(1.0, 50.0),
    notch_filter=True,
    out_file=None,
):
    return processor.create_interactive_plot(
        file_name,
        eeg_df,
        audio_io,
        gaze_df,
        window_size,
        column_range,
        ica=ica,
        bandpass=bandpass,
        notch_filter=notch_filter,
        out_file=out_file,
    )


# ---------------------------------------------------------------------------
# EEG-to-Text page
# ---------------------------------------------------------------------------

def render_eeg2text():
    try:
        from src.eeg2text_page import render_eeg2text_page
    except ImportError as exc:
        st.error(f"EEG2Text module could not be imported: {exc}")
        return

    import numpy as np

    HF_DATASET_ID = "sajjad5221/eeg2text-emmt-dataset"

    @st.cache_resource(show_spinner="Loading dataset from HuggingFace...")
    def _load_hf_dataset():
        from datasets import load_dataset
        return load_dataset(HF_DATASET_ID)

    hf_dataset = _load_hf_dataset()

    st.sidebar.success(
        "Dataset loaded: "
        + "  |  ".join(f"{s}: {len(hf_dataset[s])}" for s in hf_dataset.keys())
    )

    @st.cache_resource(show_spinner="Indexing sentences...")
    def _build_index(_ds):
        sid_index = {}
        sid_text = {}
        for split in _ds.keys():
            for idx, sample in enumerate(_ds[split]):
                sid = str(sample.get("sentence_id", "")).strip().upper()
                pid = str(sample.get("participant_id", "")).strip().upper()
                txt = str(sample.get("sentence_text", "")).strip()
                if not sid:
                    continue
                sid_index.setdefault(sid, []).append((pid, split, idx))
                if sid not in sid_text and txt:
                    sid_text[sid] = txt
        return sid_index, sid_text

    hf_index, hf_texts = _build_index(hf_dataset)

    available_sids = sorted(hf_index.keys())
    if not available_sids:
        st.error("No samples found in the HuggingFace dataset.")
        return

    sentences_df = pd.DataFrame(
        {
            "sentence_id":      available_sids,
            "sentence_content": [hf_texts.get(s, "(unavailable)") for s in available_sids],
        }
    )
    st.sidebar.info(f"Sentences available: {len(sentences_df)}")

    def _eeg_loader(sentence_id, _unused=None):
        sid = sentence_id.upper()
        candidates = hf_index.get(sid, [])
        if not candidates:
            st.warning(f"No HF samples found for {sid}.")
            dummy = np.zeros(128, dtype=np.float32)
            return {ch: dummy for ch in ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]}

        pid_options = sorted({c[0] for c in candidates})
        selected_pid = st.sidebar.selectbox(
            "Participant",
            options=pid_options,
            key=f"participant_select_{sid}",
        )

        pid, split, idx = next(
            ((p, s, i) for p, s, i in candidates if p == selected_pid),
            candidates[0],
        )
        sample = hf_dataset[split][idx]

        dummy = np.zeros(128, dtype=np.float32)
        return {
            "RAW_TP9":         dummy,
            "RAW_AF7":         dummy,
            "RAW_AF8":         dummy,
            "RAW_TP10":        dummy,
            "_sentence_id":    sid,
            "_participant_id": pid,
            "_hf_split":       split,
            "_hf_idx":         idx,
            "_hf_sample":      sample,
        }

    def _gaze_loader(_sid, _pid):
        return None

    render_eeg2text_page(
        sentences_df=sentences_df,
        eeg_data_loader=_eeg_loader,
        gaze_data_loader=_gaze_loader,
    )


# ---------------------------------------------------------------------------
# Main app function
# ---------------------------------------------------------------------------

def main():
    # Track which page to show
    if "show_eeg2text" not in st.session_state:
        st.session_state.show_eeg2text = False

    # ── EEG-to-Text page ──────────────────────────────────────────────────────
    if st.session_state.show_eeg2text:
        st.sidebar.header("Controls")
        if st.sidebar.button("Back to Data Browser"):
            st.session_state.show_eeg2text = False
            st.rerun()
        render_eeg2text()
        return

    # ── Original Data Browser (karelvlk style) ────────────────────────────────
    title_col, btn_col = st.columns([3, 1])
    with title_col:
        st.title("EEG Data Browser")
    with btn_col:
        st.write("")  # vertical alignment nudge
        st.write("")
        if st.button("🧠 EEG-to-Text", type="secondary", use_container_width=True):
            st.session_state.show_eeg2text = True
            st.rerun()

    # Sidebar for controls
    st.sidebar.header("Controls")

    # Category selection
    selected_category = shared_selection[1] if shared_selection is not None else st.sidebar.selectbox("Select Category", loader.get_categories())

    # Initialize filter in session state if not present
    if "participant_filter" not in st.session_state:
        st.session_state.participant_filter = "All"
    if "sentence_filter" not in st.session_state:
        st.session_state.sentence_filter = "All"

    # Callback functions to update filters
    def on_participant_change():
        st.session_state.sentence_filter = "All"
        # Reset file index when filter changes
        st.session_state.current_file_idx = 0

    def on_sentence_change():
        st.session_state.participant_filter = "All"
        # Reset file index when filter changes
        st.session_state.current_file_idx = 0

    # Get participant IDs with sentence filter - using cached function
    participant_ids = get_participant_ids(selected_category, st.session_state.sentence_filter)

    # Get sentence IDs with participant filter - using cached function
    sentence_ids = get_sentence_ids(selected_category, st.session_state.participant_filter)

    # Participant and Sentence filters with callbacks
    selected_participant = shared_selection[0].participant if shared_selection is not None else st.sidebar.selectbox(
        "Filter by Participant",
        participant_ids,
        key="participant_filter",
        on_change=on_participant_change,
    )

    selected_sentence = shared_selection[0].sentence if shared_selection is not None else st.sidebar.selectbox(
        "Filter by Sentence",
        sentence_ids,
        key="sentence_filter",
        on_change=on_sentence_change,
    )

    # Cache this operation
    @st.cache_data
    def get_filtered_files(category, participant, sentence):
        return loader.get_data_files(
            category_filter=category,
            participant_id_filter=(participant if participant != "All" else None),
            sentence_id_filter=(sentence if sentence != "All" else None),
        )

    # NOW we can get all files for the selected category with filters
    all_files = get_filtered_files(selected_category, selected_participant, selected_sentence)
    if shared_selection is not None:
        unit = signal_unit_for_trial(loader, *shared_selection)
        all_files = [unit] if unit is not None else []
        row, phase = shared_selection
        st.caption(f"Shared trial: {row.participant}_{row.sentence} · {row.condition} · {phase} · order {row.get('order', '')}")
        st.write("English source:", row.source)
        image_path = str(row.get("image_path", ""))
        image_file = resolve_ui(loader.base_path, image_path)
        if image_file is not None:
            st.image(str(image_file), width=350,
                     caption="Same assigned image as Translation process" + (" (not shown during this phase)" if phase in ("Read", "Translate") else ""))
        trial_audio = resolve_ui(loader.base_path, str(row.get("raw_mp3_path", "")))
        if trial_audio is not None:
            st.audio(str(trial_audio), format="audio/mpeg")
            st.caption("Same whole-trial audio as Translation process; not synchronized to this phase or the signal plot.")

    if not all_files:
        if shared_selection is not None:
            st.warning(f"{selected_category} signals are not locally cached for this exact trial. The selected person, sentence and image have not changed.")
            return
        st.warning("No EEG or gaze CSV files found with the selected filters.")
        st.info(f"Data source: {loader.base_path}. Try another phase or filter; only locally cached recordings are available in local mode.")
        return

    # Reset current file index if filters change
    filter_key = f"{selected_category}_{selected_participant}_{selected_sentence}"
    if shared_selection is not None:
        filter_key += "_" + shared_selection[0].identity
    if "filter_key" not in st.session_state or st.session_state.filter_key != filter_key:
        st.session_state.current_file_idx = 0
        st.session_state.filter_key = filter_key

    # Store current file index in session state
    if "current_file_idx" not in st.session_state:
        st.session_state.current_file_idx = 0

    # File navigation AFTER the sidebar controls
    file_names = [f"{i + 1}. {f['name']}" for i, f in enumerate(all_files)]
    nav_container = st.container()
    main_nav_col1, main_nav_col2, main_nav_col3 = nav_container.columns([1, 2, 1])

    # Use empty for more precise control of vertical space
    with main_nav_col2:
        selected_file_idx_main = st.selectbox(
            "Select File",
            range(len(file_names)),
            format_func=lambda i: file_names[i],
            index=st.session_state.current_file_idx,
            key=f"file_selector_{filter_key}_{st.session_state.current_file_idx}",
            label_visibility="collapsed",
        )
        if selected_file_idx_main != st.session_state.current_file_idx:
            st.session_state.current_file_idx = selected_file_idx_main
            st.rerun()

    # Now the buttons will be positioned after the selectbox is rendered
    with main_nav_col1:
        # Create a container at the vertical middle
        if st.button("← Previous", key="prev_main", use_container_width=True, disabled=shared_selection is not None):
            st.session_state.current_file_idx = (st.session_state.current_file_idx - 1) % len(all_files)
            st.rerun()

    with main_nav_col3:
        # Create a container at the vertical middle
        if st.button("Next →", key="next_main", use_container_width=True, disabled=shared_selection is not None):
            st.session_state.current_file_idx = (st.session_state.current_file_idx + 1) % len(all_files)
            st.rerun()

    # Update current file selection
    current_data_unit = all_files[st.session_state.current_file_idx]
    eeg_df, gaze_df, audio_io = load_file_data(current_data_unit)

    if eeg_df is None:
        st.caption(f"{current_data_unit['participant_id']} · {current_data_unit['sentence_id']} · {current_data_unit['category']}")
        published_eeg = current_data_unit.get("eeg_source_path", current_data_unit.get("eeg"))
        message = ("EEG could not be loaded for this exact trial/phase; check the file/download warning above. "
                   if published_eeg else "No preprocessed EEG is published for this exact trial/phase. ")
        st.info(message + "Gaze can be inspected independently.")
        if gaze_df is not None:
            extent = st.sidebar.selectbox("Screen coordinate range", ["1920 × 1080 (published code)", "1280 × 1024 (legacy Signals)"])
            size = (1920, 1080) if extent.startswith("1920") else (1280, 1024)
            rectangle = None
            manifest_path = os.getenv("PROCESS_MANIFEST", "")
            if current_data_unit["category"] in ("See", "Update") and Path(manifest_path).is_file():
                from PIL import Image
                manifest = pd.read_csv(manifest_path, keep_default_na=False)
                gaze_column = f"gaze_{current_data_unit['category']}_path"
                if gaze_column in manifest and "image_path" in manifest:
                    matching = manifest[manifest[gaze_column].map(lambda path: Path(path).stem == current_data_unit["name"])]
                    if len(matching) == 1:
                        image_path = Path(loader.base_path) / matching.iloc[0].image_path
                        if image_path.is_file():
                            with Image.open(image_path) as image:
                                rectangle = (650 - image.width / 2, 820 - image.height / 2, image.width, image.height)
            st.plotly_chart(processor.create_gaze_kde_heatmap(gaze_df, size, rectangle), use_container_width=True)
            if rectangle is not None:
                st.caption("Dashed rectangle in the fixation panel: assumed image bounds from published code, NOT verified against the recording. Gaze outside this rectangle does not appear in the image heatmap.")
            st.caption("This reference uses raw coordinate samples and adaptive KDE smoothing, unlike the duration-weighted image heatmap. Screen range is a display setting, not verified recording geometry. Empty panels indicate insufficient non-degenerate samples.")
        else:
            st.warning("Gaze is not locally available for this trial.")
        if audio_io is not None:
            st.audio(audio_io)
        st.sidebar.info(f"{len(all_files)} recordings match your filters")
        return

    # File information badges
    badge_category = current_data_unit["category"]
    badge_participant_id = current_data_unit["participant_id"]
    badge_sentence_id = current_data_unit["sentence_id"]
    st.markdown(
        f":violet-badge[Category: {badge_category}] :orange-badge[Participant: {badge_participant_id}] :blue-badge[Sentence: {badge_sentence_id}]"
    )
    st.caption("Signal panels currently use independent recording-relative time axes. Shared hover does not establish EEG/gaze/audio synchronization.")

    # Column selection for plotting
    st.sidebar.subheader("Plot Settings")

    # Initialize session state for checkboxes if not present
    if "compare_raw" not in st.session_state:
        st.session_state.compare_raw = False
    if "compare_raw_next_to_each_other" not in st.session_state:
        st.session_state.compare_raw_next_to_each_other = False
    if "use_ica" not in st.session_state:
        st.session_state.use_ica = False
    if "gaze_window_size" not in st.session_state:
        st.session_state.gaze_window_size = 100  # Default 100ms

    # Add the Compare Raw toggle
    st.session_state.compare_raw = st.sidebar.checkbox(
        "Compare Raw", value=st.session_state.compare_raw, key="compare_raw_cb"
    )

    st.session_state.compare_raw_next_to_each_other = st.sidebar.checkbox(
        "Plots next to each other",
        value=st.session_state.compare_raw_next_to_each_other,
        disabled=not st.session_state.compare_raw,
        key="compare_raw_next_cb",
    )

    st.session_state.use_ica = st.sidebar.checkbox("Use ICA", value=st.session_state.use_ica, key="use_ica_cb")

    # Gaze window size slider
    st.session_state.gaze_window_size = st.sidebar.slider(
        "Gaze sampling window (ms)",
        min_value=10,
        max_value=1000,
        value=st.session_state.gaze_window_size,
        step=10,
        help="Size of the time window for gaze intensity sampling in milliseconds",
    )

    # Read CSV to get column names
    all_columns = list(eeg_df.columns)

    # Allow user to select columns to plot
    start_col = st.sidebar.selectbox("Start Column", all_columns, index=min(21, len(all_columns) - 1))
    end_col = st.sidebar.selectbox("End Column", all_columns, index=min(24, len(all_columns) - 1))

    start_idx = all_columns.index(start_col)
    end_idx = all_columns.index(end_col) + 1  # +1 for inclusive range

    # Plot the data using cached function
    figures = process_and_plot_data_cached(
        current_data_unit["name"],
        eeg_df,
        audio_io,
        gaze_df,
        st.session_state.gaze_window_size / 1000.0,  # Convert ms to seconds
        (start_idx, end_idx),
        ica=st.session_state.use_ica,
    )

    if isinstance(figures, tuple):
        main_fig, heatmap_fig = figures
        st.plotly_chart(main_fig, use_container_width=True)
        if heatmap_fig is not None:
            st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.plotly_chart(figures, use_container_width=True)

    if st.session_state.compare_raw:
        raw_figures = process_and_plot_data_cached(
            current_data_unit["name"],
            eeg_df,
            audio_io,
            gaze_df,
            st.session_state.gaze_window_size / 1000.0,
            (start_idx, end_idx),
            ica=None,
            bandpass=None,
            notch_filter=None,
            out_file=None,
        )

        if st.session_state.compare_raw_next_to_each_other:
            col1, col2 = st.columns(2)
            with col1:
                if isinstance(figures, tuple):
                    st.plotly_chart(figures[0], use_container_width=True)
                else:
                    st.plotly_chart(figures, use_container_width=True)
            with col2:
                if isinstance(raw_figures, tuple):
                    st.plotly_chart(raw_figures[0], use_container_width=True)
                else:
                    st.plotly_chart(raw_figures, use_container_width=True)
        else:
            if isinstance(figures, tuple):
                st.plotly_chart(figures[0], use_container_width=True)
            else:
                st.plotly_chart(figures, use_container_width=True)
            if isinstance(raw_figures, tuple):
                st.plotly_chart(raw_figures[0], use_container_width=True)
            else:
                st.plotly_chart(raw_figures, use_container_width=True)

    if audio_io is not None:
        st.audio(audio_io)
    else:
        st.warning("No audio file found")

    # Show data preview
    st.subheader("EEG Data Preview")
    st.dataframe(eeg_df.head())

    # Create dataframe with all filtered files - using cached function
    files_df = create_files_dataframe(all_files)
    st.sidebar.dataframe(
        files_df[["Filename", "Category", "Participant", "Sentence"]],
        use_container_width=True,
        height=600,
    )

    # Add info about number of files
    st.sidebar.info(f"{len(all_files)} files match your filters")


if __name__ == "__main__":
    main()
