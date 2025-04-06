import streamlit as st
import pandas as pd
from pathlib import Path
import re
import processor

# Set page configuration
st.set_page_config(page_title="EEG Data Browser", layout="wide")

# Define paths
BASE_DATA_DIR = Path("./ufal_emmt/preprocessed-data/eeg")
CATEGORIES = ["Read", "See", "Update", "Translate", "All"]


def extract_participant_id(filename):
    """Extract participant ID (P<number>) from filename."""
    match = re.search(r"P\d+", filename)
    return match.group(0) if match else None


def extract_sentence_id(filename):
    """Extract sentence ID (S<number>) from filename."""
    match = re.search(r"S\d+", filename)
    return match.group(0) if match else None


def get_csv_files(category, participant_filter=None, sentence_filter=None):
    """
    Get all CSV files for a given category or all categories,
    optionally filtered by participant ID and sentence ID.
    """
    if category == "All":
        files = []
        for cat in CATEGORIES[:-1]:  # Exclude "All" from the list
            cat_path = BASE_DATA_DIR / cat
            if cat_path.exists():
                files.extend(list(cat_path.glob("*.csv")))
    else:
        category_path = BASE_DATA_DIR / category
        if category_path.exists():
            files = list(category_path.glob("*.csv"))
        else:
            return []

    # Apply filters if provided
    if participant_filter and participant_filter != "All":
        files = [f for f in files if extract_participant_id(f.name) == participant_filter]

    if sentence_filter and sentence_filter != "All":
        files = [f for f in files if extract_sentence_id(f.name) == sentence_filter]

    return files


def get_unique_identifiers(category, id_extractor, participant_filter=None, sentence_filter=None):
    """
    Get unique participant or sentence IDs from files in a category.
    Can be filtered by participant or sentence ID.
    """
    # Get all files first, potentially filtered
    files = get_csv_files(category, participant_filter, sentence_filter)

    # Extract IDs from filenames
    ids = [id_extractor(f.name) for f in files]
    # Filter out None values and return unique IDs
    return sorted(list(set(id for id in ids if id is not None)))


def create_files_dataframe(files):
    """
    Create a dataframe with information about all files that match the current filters.

    Args:
        files: List of Path objects representing CSV files

    Returns:
        DataFrame with columns: Filename, Category, Participant, Sentence
    """
    data = []
    for file in files:
        filename = file.name
        category = file.parts[-2]  # The parent directory is the category
        participant_id = extract_participant_id(filename) or "N/A"
        sentence_id = extract_sentence_id(filename) or "N/A"

        data.append(
            {
                "Filename": filename,
                "Category": category,
                "Participant": participant_id,
                "Sentence": sentence_id,
                "Path": str(file),  # Store the full path for reference
            }
        )

    return pd.DataFrame(data)


# Main app function
def main():
    st.title("EEG Data Browser")

    # Sidebar for controls
    st.sidebar.header("Controls")

    # Category selection
    selected_category = st.sidebar.selectbox("Select Category", CATEGORIES)

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

    # Get participant IDs with sentence filter
    participant_ids = ["All"] + get_unique_identifiers(
        selected_category,
        extract_participant_id,
        sentence_filter=(st.session_state.sentence_filter if st.session_state.sentence_filter != "All" else None),
    )

    # Get sentence IDs with participant filter
    sentence_ids = ["All"] + get_unique_identifiers(
        selected_category,
        extract_sentence_id,
        participant_filter=(
            st.session_state.participant_filter if st.session_state.participant_filter != "All" else None
        ),
    )

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

    # NOW we can get all files for the selected category with filters
    all_files = get_csv_files(
        selected_category,
        participant_filter=(selected_participant if selected_participant != "All" else None),
        sentence_filter=selected_sentence if selected_sentence != "All" else None,
    )

    if not all_files:
        st.warning(f"No CSV files found with the selected filters.")
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
    file_names = [f"{i+1}. {f.name}" for i, f in enumerate(all_files)]
    nav_container = st.container()
    main_nav_col1, main_nav_col2, main_nav_col3 = nav_container.columns([1, 2, 1])

    # Use empty for more precise control of vertical space
    with main_nav_col2:
        selected_file_idx_main = st.selectbox(
            "",  # Empty label
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
    current_file = all_files[st.session_state.current_file_idx]

    # File information badges
    badge_category = current_file.parts[-2]
    badge_participant_id = extract_participant_id(current_file.name) or "N/A"
    badge_sentence_id = extract_sentence_id(current_file.name) or "N/A"
    st.markdown(
        f":violet-badge[Category: {badge_category}] :orange-badge[Participant: {badge_participant_id}] :blue-badge[Sentence: {badge_sentence_id}]"
    )

    # Column selection for plotting
    st.sidebar.subheader("Plot Settings")

    # Add the Compare Raw toggle
    compare_raw = st.sidebar.checkbox("Compare Raw", value=False)
    compare_raw_next_to_each_other = st.sidebar.checkbox(
        "Plots next to each other", value=False, disabled=not compare_raw
    )

    # Read CSV to get column names
    df = pd.read_csv(current_file)
    all_columns = list(df.columns)

    # Allow user to select columns to plot
    start_col = st.sidebar.selectbox("Start Column", all_columns, index=min(21, len(all_columns) - 1))
    end_col = st.sidebar.selectbox("End Column", all_columns, index=min(24, len(all_columns) - 1))

    start_idx = all_columns.index(start_col)
    end_idx = all_columns.index(end_col) + 1  # +1 for inclusive range

    # Plot the data
    if compare_raw:
        # Process and plot both raw and processed data
        raw_fig, fig = processor.process_and_plot_eeg_data(current_file, (start_idx, end_idx), compare_raw=True)
        if compare_raw_next_to_each_other:
            image_col1, image_col2 = st.columns(2)
            with image_col1:
                st.pyplot(fig)
            with image_col2:
                st.pyplot(raw_fig)
        else:
            st.pyplot(fig)
            st.pyplot(raw_fig)
    else:
        # Only process and plot the processed data
        fig = processor.process_and_plot_eeg_data(current_file, (start_idx, end_idx), compare_raw=False)
        st.pyplot(fig)

    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Move the DataFrame display to the sidebar
    st.sidebar.subheader("Files List")

    # Create dataframe with all filtered files
    files_df = create_files_dataframe(all_files)

    # Highlight the currently selected row
    current_file_path = str(current_file)

    # Style the dataframe to highlight the selected row
    def highlight_selected_row(row):
        if row["Path"] == current_file_path:
            return ["background-color: #ffffb3"] * len(row)
        return [""] * len(row)

    # Apply styling and display dataframe (excluding the Path column)
    styled_df = files_df.style.apply(highlight_selected_row, axis=1)
    st.sidebar.dataframe(
        styled_df.data[["Filename", "Category", "Participant", "Sentence"]],
        use_container_width=True,
        height=600,
    )

    # Add info about number of files
    st.sidebar.info(f"{len(all_files)} files match your filters")


if __name__ == "__main__":
    main()
