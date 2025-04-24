import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from data_preprocessing import preprocess_raw_data
import streamlit as st


@st.cache_data(show_spinner=False)
def get_data(csv_file, cols, **kwargs):
    return preprocess_raw_data(csv_file, cols, **kwargs)


def get_related_files(csv_file: str) -> tuple[str | None, str | None]:
    raw_basename = os.path.basename(csv_file).split(".")[0]
    p, o, s, c1, c2 = raw_basename.split("-")
    # :facepalm: i don't fucking know what the fuck is happening during data preprocessing
    basename = f"{p}-{str(int(o)-1).zfill(2)}-{s}-{c1}-{c2}"

    category = os.path.basename(os.path.dirname(csv_file))

    print("base", basename)
    print("category", category)

    audio_file = os.path.join(os.path.dirname(csv_file), "../../", "audio", category, f"{basename}.wav")
    if not os.path.exists(audio_file):
        logging.warning(f"Audio file not found: {audio_file}")
        audio_file = None

    gaze_file = os.path.join(os.path.dirname(csv_file), "../../", "gaze", category, f"{basename}.csv")
    if not os.path.exists(gaze_file):
        logging.warning(f"Gaze file not found: {gaze_file}")
        gaze_file = None

    return audio_file, gaze_file


def process_and_plot_eeg_data(
    csv_file: str,
    cols: tuple[int, int] = (21, 25),
    method: str = "both",  # "eeg", "audio", or "both"
    **kwargs,
) -> tuple[plt.Figure, str | None, str | None]:
    """
    Plot EEG time series data and/or audio from a CSV file.
    Args:
        csv_file: Path to the CSV file
        cols: Tuple of column indices to use
        method: What to plot - "eeg", "audio", or "both"
        **kwargs: Additional arguments passed to get_data
    Returns:
        Figure for Streamlit to display
    """
    audio_file, gaze_file = get_related_files(csv_file)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    if method in ["eeg", "both"]:
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
                ax.plot(df["Time"], df[col], label=col)
        else:
            logging.warning("No EEG data available")

    if method in ["audio", "both"] and audio_file:
        try:
            # Load audio file
            y, sr = librosa.load(audio_file)
            audio_time = np.linspace(0, len(y) / sr, len(y))

            if method == "both" and "eeg_columns" in locals():
                # Normalize audio to match EEG scale and shift down
                y_normalized = y * (np.max(df[eeg_columns].values) - np.min(df[eeg_columns].values)) / 2
                y_normalized = y_normalized - 200  # Shift down to y=-200
                ax.plot(audio_time, y_normalized, color="gray", alpha=0.3, label="Audio")
            else:
                # Plot raw audio shifted down
                y_shifted = y - 200  # Shift down to y=-200
                ax.plot(audio_time, y_shifted, color="gray", alpha=0.3, label="Audio")
        except Exception as e:
            logging.warning(f"Failed to load audio file: {e}")
    elif method in ["audio", "both"]:
        logging.warning("No audio file found")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal Value")
    title = {"eeg": "EEG Time Series", "audio": "Audio Waveform", "both": "EEG Time Series with Audio"}[method]
    ax.set_title(f"{title} - {csv_file}")
    ax.legend()
    ax.grid(True)

    return fig, audio_file, gaze_file


def plot_raw_eeg_data(csv_file: str, cols: tuple[int, int] = (21, 25)) -> tuple[plt.Figure, str | None, str | None]:
    """
    Plot EEG time series data from a CSV file.
    Returns the figure for Streamlit to display.
    """
    return process_and_plot_eeg_data(csv_file, cols, method="eeg", ica=False, notch_filter=None, bandpass=None)
