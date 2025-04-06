import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_raw_data
import streamlit as st


def process_and_plot_eeg_data(
    csv_file: str, cols: tuple[int, int] = (21, 25), **kwargs
) -> tuple[plt.Figure, plt.Figure] | plt.Figure:
    """
    Plot EEG time series data from a CSV file.
    Returns the figure for Streamlit to display.
    """
    mne_raw = preprocess_raw_data(csv_file, cols, **kwargs)

    # Convert MNE Raw data back to Pandas DataFrame
    data = mne_raw.get_data()  # Get the EEG data
    time = mne_raw.times  # Get the time vector
    eeg_columns = mne_raw.ch_names  # Get the channel names

    # Create a DataFrame
    df = pd.DataFrame(data.T, columns=eeg_columns)  # Transpose to have channels as columns
    df["Time"] = time  # Add time as a column

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in eeg_columns:
        ax.plot(df["Time"], df[col], label=col)  # Plot using the DataFrame

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EEG Signal Value")
    ax.set_title(f"EEG Time Series - {csv_file}")
    ax.legend()
    ax.grid(True)

    return fig


def plot_raw_eeg_data(csv_file: str, cols: tuple[int, int] = (21, 25)) -> plt.Figure:
    """
    Plot EEG time series data from a CSV file.
    Returns the figure for Streamlit to display.
    """
    return process_and_plot_eeg_data(csv_file, cols, ica=False, notch_filter=None, bandpass=None)
