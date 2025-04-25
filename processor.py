import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from data_preprocessing import preprocess_raw_data
import streamlit as st
import seaborn as sns


@st.cache_data(show_spinner=False)
def get_data(csv_file, cols, **kwargs):
    return preprocess_raw_data(csv_file, cols, **kwargs)


def get_related_files(csv_file: str) -> tuple[str | None, str | None]:
    raw_basename = os.path.basename(csv_file).split(".")[0]
    p, o, s, c1, c2 = raw_basename.split("-")
    # :facepalm: i don't fucking know what the fuck is happening during data preprocessing
    basename_audio = f"{p}-{str(int(o)-1).zfill(2)}-{s}-{c1}-{c2}"
    basename_gaze = f"{p}-{o}-{s}-{c1}-{c2}"

    category = os.path.basename(os.path.dirname(csv_file))

    audio_file = os.path.join(os.path.dirname(csv_file), "../../", "audio", category, f"{basename_audio}.wav")
    if not os.path.exists(audio_file):
        logging.warning(f"Audio file not found: {audio_file}")
        audio_file = None

    gaze_file = os.path.join(os.path.dirname(csv_file), "../../", "gaze", category, f"{basename_gaze}.csv")
    if not os.path.exists(gaze_file):
        logging.warning(f"Gaze file not found: {gaze_file}")
        gaze_file = None

    return audio_file, gaze_file


def plot_gaze_heatmap(df: pd.DataFrame) -> plt.Figure:
    # Ensure X and Y are numeric
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    # Drop rows with invalid coordinates
    df = df.dropna(subset=["X", "Y"])
    # Rozdělení podle typu události
    saccade_df = df[df["Event"] == "saccade"]
    fixation_df = df[df["Event"] == "fixation"]

    # Společný rozsah os
    x_min = 0  # vynucený začátek na 0
    x_max = 1280
    y_min = 0
    y_max = 1024

    # Vytvoření grafu
    fig, axes = plt.subplots(1, 3, figsize=(12, 18))

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Osa Y obráceně (0,0 vlevo nahoře)
        ax.set_aspect("equal")  # Stejné měřítko

    # Heatmapa Saccade
    sns.kdeplot(x=saccade_df["X"], y=saccade_df["Y"], cmap="Reds", fill=True, ax=axes[0])
    axes[0].set_title("Heatmap of Saccade Events")

    # Heatmapa Fixation
    sns.kdeplot(x=fixation_df["X"], y=fixation_df["Y"], cmap="Blues", fill=True, ax=axes[1])
    axes[1].set_title("Heatmap of Fixation Events")

    # Kombinovaná heatmapa
    sns.kdeplot(x=saccade_df["X"], y=saccade_df["Y"], cmap="Reds", fill=True, alpha=0.5, ax=axes[2], label="Saccade")
    sns.kdeplot(
        x=fixation_df["X"], y=fixation_df["Y"], cmap="Blues", fill=True, alpha=0.5, ax=axes[2], label="Fixation"
    )
    axes[2].set_title("Combined Heatmap of Events")
    axes[2].legend()

    return fig


@st.cache_data(show_spinner=True)
def process_and_plot_eeg_data(
    csv_file: str,
    cols: tuple[int, int] = (21, 25),
    plot_eeg: bool = True,
    plot_audio: bool = True,
    plot_gaze: bool = True,
    **kwargs,
) -> tuple[plt.Figure, str | None, plt.Figure | None]:
    """
    Plot EEG time series data and/or audio from a CSV file.
    Args:
        csv_file: Path to the CSV file
        cols: Tuple of column indices to use
        method: What to plot - "eeg", "audio", or "both"
        plot_eeg: Whether to plot EEG data
        plot_audio: Whether to plot audio data
        plot_gaze: Whether to plot gaze data
        **kwargs: Additional arguments passed to get_data
    Returns:
        Figure for Streamlit to display
    """
    audio_file, gaze_file = get_related_files(csv_file)

    # Create a single plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define vertical offsets for different signals
    eeg_offset = 0
    audio_offset = -200

    # Define colors for different signals
    audio_color = "gray"
    gaze_color = "red"

    if plot_eeg:
        mne_raw = get_data(csv_file, cols, **kwargs)
        if mne_raw is not None:
            # Convert MNE Raw data back to Pandas DataFrame
            data = mne_raw.get_data()  # Get the EEG data
            time = mne_raw.times  # Get the time vector
            eeg_columns = mne_raw.ch_names  # Get the channel names

            # Create a DataFrame
            df = pd.DataFrame(data.T, columns=eeg_columns)  # Transpose to have channels as columns
            df["Time"] = time  # Add time as a column

            # Plot EEG data
            for col in eeg_columns:
                ax.plot(df["Time"], df[col] + eeg_offset, label=col)
        else:
            logging.warning("No EEG data available")

    if plot_audio and audio_file:
        try:
            # Load audio file
            y, sr = librosa.load(audio_file)
            audio_time = np.linspace(0, len(y) / sr, len(y))

            if plot_eeg or plot_gaze:
                # Normalize audio to match EEG scale
                y_normalized = y * (np.max(df[eeg_columns].values) - np.min(df[eeg_columns].values)) / 2
                ax.plot(audio_time, y_normalized + audio_offset, color=audio_color, alpha=0.3, label="Audio")
            else:
                # Plot raw audio
                ax.plot(audio_time, y + audio_offset, color=audio_color, alpha=0.3, label="Audio")
        except Exception as e:
            logging.warning(f"Failed to load audio file and process it: {e}")

    # Plot gaze intensity if gaze file exists
    gaze_heatmap = None
    if plot_gaze and gaze_file:
        try:
            # --- Load gaze data ---
            gaze_df = pd.read_csv(gaze_file)
            gaze_heatmap = plot_gaze_heatmap(gaze_df)
            # Parse numeric X/Y and handle blinks ("." becomes NaN)
            gaze_df["X"] = pd.to_numeric(gaze_df["X"], errors="coerce")
            gaze_df["Y"] = pd.to_numeric(gaze_df["Y"], errors="coerce")

            # Parse TimeStamp and remove rows with invalid data
            gaze_df["TimeDT"] = pd.to_datetime(gaze_df["TimeStamp"], format="%H:%M:%S.%f", errors="coerce")
            gaze_df = gaze_df.dropna(subset=["TimeDT", "X", "Y"])  # Skip blinks

            # Compute elapsed time
            gaze_df["Time"] = (gaze_df["TimeDT"] - gaze_df["TimeDT"].min()).dt.total_seconds()

            # Compute delta movement (absolute change)
            gaze_df["deltaX"] = gaze_df["X"].diff().abs()
            gaze_df["deltaY"] = gaze_df["Y"].diff().abs()
            gaze_df["Movement"] = gaze_df["deltaX"] + gaze_df["deltaY"]

            # Bin into time windows
            window_size = 0.1  # seconds
            max_time = gaze_df["Time"].max()
            bins = np.arange(0, max_time + window_size, window_size)
            gaze_df["bin"] = np.floor(gaze_df["Time"] / window_size).astype(int)

            # Sum movement per time window
            movement_per_bin = gaze_df.groupby("bin")["Movement"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)

            # Normalize to 0–100 scale
            scale = 100.0 / max(movement_per_bin.max(), 1)
            movement_scaled = movement_per_bin * scale

            # Offset and plot
            ax.bar(
                bins[:-1],
                movement_scaled,
                width=window_size,
                alpha=0.6,
                label="Gaze intensity (ΔX+ΔY)",
                color=gaze_color,
                bottom=-400,
            )

            # Format x-axis
            def format_time(sec):
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            tick_interval = 0.5
            ticks = np.arange(0, max_time + tick_interval, tick_interval)
            ax.set_xticks(ticks)
            ax.set_xticklabels([format_time(t) for t in ticks])

            # # Plot vertical lines for each Event (if present)
            if "Event" in gaze_df.columns:
                event_df = gaze_df.dropna(subset=["Event", "Time"])
                unique_events = event_df["Event"].unique()
                event_colors = plt.cm.tab10.colors  # Up to 10 distinct colors
                color_map = {event: event_colors[i % len(event_colors)] for i, event in enumerate(unique_events)}

                for event in unique_events:
                    times = event_df.loc[event_df["Event"] == event, "Time"].values
                    ax.vlines(
                        x=times,
                        ymin=-410,
                        ymax=-405,
                        colors=color_map[event],
                        linestyles="-",
                        linewidth=1,
                        label=f"Event: {event}",
                    )

                # Deduplicate legend entries
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

        except Exception as e:
            logging.warning(f"Failed to load gaze file and process it: {e}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal Value")

    # Build title based on what's being plotted
    title_parts = []
    if plot_eeg:
        title_parts.append("EEG")
    if plot_audio:
        title_parts.append("Audio")
    if plot_gaze:
        title_parts.append("Gaze")

    title = " + ".join(title_parts) if title_parts else "No Data Selected"
    ax.set_title(f"{title} - {csv_file}")

    ax.legend(
        fontsize="small",  # Reduce font size
        markerscale=0.7,  # Smaller legend markers
        handlelength=1.0,  # Length of the legend line
        handletextpad=0.5,  # Space between legend marker and text
        labelspacing=0.3,  # Vertical space between labels
        borderpad=0.3,  # Padding inside the legend box
        borderaxespad=0.3,  # Padding between the axes and legend box
        loc="best",  # Auto position
    )
    ax.grid(True)

    return fig, audio_file, gaze_heatmap


def plot_raw_eeg_data(
    csv_file: str, cols: tuple[int, int] = (21, 25)
) -> tuple[plt.Figure, str | None, plt.Figure | None]:
    """
    Plot EEG time series data from a CSV file.
    Returns the figure for Streamlit to display.
    """
    return process_and_plot_eeg_data(csv_file, cols, method="eeg", ica=False, notch_filter=None, bandpass=None)
