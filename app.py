import os
import streamlit as st
import src.processor as processor
import pandas as pd
from src.file_loader import GithubFileLoader, FileLoader

# Set page configuration
st.set_page_config(page_title="EEG Data Browser", layout="wide")


# Singleton implementation for loader
@st.cache_resource
def get_loader():
    file_loader_type = os.getenv("FILE_LOADER_TYPE")
    data_path = os.getenv("DATA_PATH")
    print(f"file_loader_type: {file_loader_type}")
    print(f"data_path: {data_path}")
    if file_loader_type == "local":
        return FileLoader(base_path=data_path)
    elif file_loader_type == "github":
        return GithubFileLoader(base_url=data_path)
    else:
        raise ValueError(f"Invalid file loader type: {file_loader_type}")


# Get the singleton instance
loader = get_loader()


@st.cache_data
def create_files_dataframe(data_units: list[dict]):
    """
    Create a dataframe with information about all files that match the current filters.

    Args:
        files: List of Path objects representing CSV files

    Returns:
        DataFrame with columns: Filename, Category, Participant, Sentence
    """
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
    return processor.process_and_plot_data(
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


@st.cache_data
def prepare_interactive_data_cached(
    file_name,
    eeg_df,
    audio_io,
    gaze_df,
    window_size,
    column_range,
    ica=None,
    bandpass=(1.0, 50.0),
    notch_filter=True,
):
    return processor.prepare_interactive_data(
        file_name,
        eeg_df,
        audio_io,
        gaze_df,
        window_size,
        column_range,
        ica=ica,
        bandpass=bandpass,
        notch_filter=notch_filter,
    )


# Main app function
def main():
    st.title("EEG Data Browser")

    # Sidebar for controls
    st.sidebar.header("Controls")

    # Category selection
    selected_category = st.sidebar.selectbox("Select Category", loader.get_categories())

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
    selected_participant = st.sidebar.selectbox(
        "Filter by Participant",
        participant_ids,
        key="participant_filter",
        on_change=on_participant_change,
    )

    selected_sentence = st.sidebar.selectbox(
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

    if not all_files:
        st.warning("No CSV files found with the selected filters.")
        return

    # Reset current file index if filters change
    filter_key = f"{selected_category}_{selected_participant}_{selected_sentence}"
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
            key="file_selector_main",
            label_visibility="collapsed",
        )
        if selected_file_idx_main != st.session_state.current_file_idx:
            st.session_state.current_file_idx = selected_file_idx_main
            st.rerun()

    # Now the buttons will be positioned after the selectbox is rendered
    with main_nav_col1:
        # Create a container at the vertical middle
        if st.button("← Previous", key="prev_main", use_container_width=True):
            st.session_state.current_file_idx = (st.session_state.current_file_idx - 1) % len(all_files)
            st.rerun()

    with main_nav_col3:
        # Create a container at the vertical middle
        if st.button("Next →", key="next_main", use_container_width=True):
            st.session_state.current_file_idx = (st.session_state.current_file_idx + 1) % len(all_files)
            st.rerun()

    # Update current file selection
    current_data_unit = all_files[st.session_state.current_file_idx]
    eeg_df, gaze_df, audio_io = load_file_data(current_data_unit)

    if eeg_df is None:
        st.warning("No EEG data found")
        return

    # File information badges
    badge_category = current_data_unit["category"]
    badge_participant_id = current_data_unit["participant_id"]
    badge_sentence_id = current_data_unit["sentence_id"]
    st.markdown(
        f":violet-badge[Category: {badge_category}] :orange-badge[Participant: {badge_participant_id}] :blue-badge[Sentence: {badge_sentence_id}]"
    )

    # Column selection for plotting
    st.sidebar.subheader("Plot Settings")

    # Initialize session state for checkboxes and options if not present
    if "use_legacy_view" not in st.session_state:
        st.session_state.use_legacy_view = False
    if "compare_raw" not in st.session_state:
        st.session_state.compare_raw = False
    if "compare_raw_next_to_each_other" not in st.session_state:
        st.session_state.compare_raw_next_to_each_other = False
    if "use_ica" not in st.session_state:
        st.session_state.use_ica = False
    if "gaze_window_size" not in st.session_state:
        st.session_state.gaze_window_size = 100  # Default 100ms
    if "show_eeg" not in st.session_state:
        st.session_state.show_eeg = True
    if "show_audio" not in st.session_state:
        st.session_state.show_audio = True
    if "show_gaze" not in st.session_state:
        st.session_state.show_gaze = True
    if "show_events" not in st.session_state:
        st.session_state.show_events = True
    if "show_gaze_heatmap" not in st.session_state:
        st.session_state.show_gaze_heatmap = True

    # View mode toggle
    st.session_state.use_legacy_view = st.sidebar.checkbox("Use Legacy View", value=st.session_state.use_legacy_view)

    # Add the Compare Raw toggle (only relevant for legacy view)
    if st.session_state.use_legacy_view:
        st.session_state.compare_raw = st.sidebar.checkbox(
            "Compare Raw", value=st.session_state.compare_raw, key="compare_raw_cb"
        )

        st.session_state.compare_raw_next_to_each_other = st.sidebar.checkbox(
            "Plots next to each other",
            value=st.session_state.compare_raw_next_to_each_other,
            disabled=not st.session_state.compare_raw,
            key="compare_raw_next_cb",
        )

    # Common settings
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

    # Interactive view show/hide options
    if not st.session_state.use_legacy_view:
        st.sidebar.subheader("Show/Hide Components")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            st.session_state.show_eeg = st.checkbox("EEG", value=st.session_state.show_eeg)
            st.session_state.show_audio = st.checkbox("Audio", value=st.session_state.show_audio)

        with col2:
            st.session_state.show_gaze = st.checkbox("Gaze", value=st.session_state.show_gaze)
            st.session_state.show_events = st.checkbox("Events", value=st.session_state.show_events)
            st.session_state.show_gaze_heatmap = st.checkbox("Gaze Heatmap", value=st.session_state.show_gaze_heatmap)

    # Process data based on view mode
    if st.session_state.use_legacy_view:
        # Legacy view - using matplotlib plots
        fig, wav, gaze_heatmap = process_and_plot_data_cached(
            current_data_unit["name"],
            eeg_df,
            audio_io,
            gaze_df,
            st.session_state.gaze_window_size / 1000.0,  # Convert ms to seconds
            (start_idx, end_idx),
            ica=st.session_state.use_ica,
        )

        if st.session_state.compare_raw:
            raw_fig, _, _ = process_and_plot_data_cached(
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
                image_col1, image_col2 = st.columns(2)
                with image_col1:
                    st.pyplot(fig)
                with image_col2:
                    st.pyplot(raw_fig)
            else:
                st.pyplot(fig)
                st.pyplot(raw_fig)
        else:
            st.pyplot(fig)

        if wav is not None:
            st.audio(wav)
        else:
            st.warning("No audio file found")

        if gaze_heatmap is not None:
            st.pyplot(gaze_heatmap)
        else:
            st.warning("No gaze heatmap data found")
    else:
        # Interactive view - using Altair for interactive charts
        processed_data = prepare_interactive_data_cached(
            current_data_unit["name"],
            eeg_df,
            audio_io,
            gaze_df,
            st.session_state.gaze_window_size / 1000.0,  # Convert ms to seconds
            (start_idx, end_idx),
            ica=st.session_state.use_ica,
        )

        # Show EEG data if available and user wants to see it
        if st.session_state.show_eeg and processed_data["eeg_data"] is not None:
            eeg_chart = processor.plot_interactive_eeg(
                processed_data["eeg_data"], processed_data["eeg_channels"], x_min=0.0, x_max=processed_data["max_time"]
            )
            st.altair_chart(eeg_chart, use_container_width=True)

        # Show audio if available and user wants to see it
        if st.session_state.show_audio and processed_data["audio_data"] is not None:
            audio_chart = processor.plot_interactive_audio(
                processed_data["audio_data"], x_min=0.0, x_max=processed_data["max_time"]
            )
            st.altair_chart(audio_chart, use_container_width=True)

        # Show gaze movement if available and user wants to see it
        if st.session_state.show_gaze and processed_data["gaze_movement_data"] is not None:
            gaze_chart = processor.plot_interactive_gaze_movement(
                processed_data["gaze_movement_data"], x_min=0.0, x_max=processed_data["max_time"]
            )
            st.altair_chart(gaze_chart, use_container_width=True)

        # Show events if available and user wants to see them
        if st.session_state.show_events and processed_data.get("event_data") is not None:
            events_chart = processor.plot_interactive_events(
                processed_data["event_data"], x_min=0.0, x_max=processed_data["max_time"]
            )
            if events_chart:
                st.altair_chart(events_chart, use_container_width=True)

        # Also provide playback control
        if audio_io is not None:
            st.audio(audio_io)

        # Show gaze heatmap if available and user wants to see it
        if st.session_state.show_gaze_heatmap and gaze_df is not None and len(gaze_df) > 0:
            gaze_heatmap = processor.plot_gaze_heatmap(gaze_df)
            st.pyplot(gaze_heatmap)

    # Show data preview
    with st.expander("EEG Data Preview", expanded=False):
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
