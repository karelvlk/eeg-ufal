import os
import io
import logging
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import streamlit as st
import seaborn as sns
import altair as alt
from src.data_preprocessing import preprocess_raw_data
from typing import Dict, List, Optional, Union, Any


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

    return fig


def prepare_interactive_data(
    name: str,
    eeg_df: pd.DataFrame | None,
    audio_io: io.BytesIO | None,
    gaze_df: pd.DataFrame | None,
    gaze_window_size: float = 0.1,  # in seconds
    cols: tuple[int, int] = (21, 25),
    **kwargs,
) -> Dict[str, Any]:
    """
    Prepare data for interactive visualization in Streamlit.

    Returns a dictionary with all processed data for plotting.
    """
    result: Dict[str, Any] = {
        "eeg_data": None,
        "audio_data": None,
        "gaze_data": None,
        "event_data": None,
        "gaze_movement_data": None,
        "name": name,
        "eeg_channels": [],
        "max_time": 0.0,  # Will store the maximum time across all data sources
    }

    # Process EEG data
    if eeg_df is not None and len(eeg_df) > 1:
        mne_raw = preprocess_raw_data(eeg_df, cols, **kwargs)
        if mne_raw is not None:
            # Convert MNE Raw data back to Pandas DataFrame
            data = mne_raw.get_data()  # Get the EEG data
            time = mne_raw.times  # Get the time vector
            eeg_columns = mne_raw.ch_names  # Get the channel names

            # Create a DataFrame with long format for Altair
            eeg_long_df = pd.DataFrame()
            for i, channel in enumerate(eeg_columns):
                channel_df = pd.DataFrame({"Time": time, "Value": data[i], "Channel": channel})
                eeg_long_df = pd.concat([eeg_long_df, channel_df])

            result["eeg_data"] = eeg_long_df
            result["eeg_channels"] = eeg_columns
            result["max_time"] = max(result["max_time"], time.max())

    # Process audio data
    if audio_io is not None:
        try:
            # Reset the file pointer to the beginning
            audio_io.seek(0)

            # Load audio file
            y, sr = librosa.load(audio_io)
            audio_time = np.linspace(0, len(y) / sr, len(y))

            # Create DataFrame for audio data
            audio_df = pd.DataFrame({"Time": audio_time, "Value": y})

            result["audio_data"] = audio_df
            result["sample_rate"] = sr
            result["max_time"] = max(result["max_time"], audio_time.max())
        except Exception as e:
            logging.warning(f"Failed to load audio file and process it: {e}")

    # Process gaze data
    if gaze_df is not None and len(gaze_df) > 1:
        try:
            print("Input gaze data columns:", gaze_df.columns.tolist())
            print("Input gaze data sample:", gaze_df.head())

            # Parse numeric X/Y and handle blinks ("." becomes NaN)
            gaze_df["X"] = pd.to_numeric(gaze_df["X"], errors="coerce")
            gaze_df["Y"] = pd.to_numeric(gaze_df["Y"], errors="coerce")

            # Parse TimeStamp and remove rows with invalid data
            gaze_df["TimeDT"] = pd.to_datetime(gaze_df["TimeStamp"], format="%H:%M:%S.%f", errors="coerce")
            gaze_df = gaze_df.dropna(subset=["TimeDT", "X", "Y"])  # Skip blinks

            # Create an explicit copy to avoid the SettingWithCopyWarning
            gaze_df = gaze_df.copy()

            # Compute elapsed time
            gaze_df.loc[:, "Time"] = (gaze_df["TimeDT"] - gaze_df["TimeDT"].min()).dt.total_seconds()

            # Compute delta movement (absolute change)
            gaze_df.loc[:, "deltaX"] = gaze_df["X"].diff().abs()
            gaze_df.loc[:, "deltaY"] = gaze_df["Y"].diff().abs()
            gaze_df.loc[:, "Movement"] = gaze_df["deltaX"] + gaze_df["deltaY"]

            # Ensure all values are positive
            gaze_df.loc[:, "deltaX"] = gaze_df["deltaX"].abs()
            gaze_df.loc[:, "deltaY"] = gaze_df["deltaY"].abs()
            gaze_df.loc[:, "Movement"] = gaze_df["Movement"].abs()

            # Bin into time windows
            window_size = gaze_window_size  # seconds
            max_time = gaze_df["Time"].max()
            bins = np.arange(0, max_time + window_size, window_size)
            gaze_df.loc[:, "bin"] = np.floor(gaze_df["Time"] / window_size).astype(int)

            # Sum movement per time window
            movement_per_bin = gaze_df.groupby("bin")["Movement"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)
            deltaX_per_bin = gaze_df.groupby("bin")["deltaX"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)
            deltaY_per_bin = gaze_df.groupby("bin")["deltaY"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)

            # Scale the summed values to 0-100 range
            max_movement = movement_per_bin.max()
            if max_movement > 0:
                movement_per_bin = (movement_per_bin / max_movement) * 100
                deltaX_per_bin = (deltaX_per_bin / deltaX_per_bin.max()) * 100
                deltaY_per_bin = (deltaY_per_bin / deltaY_per_bin.max()) * 100

            # Prepare gaze movement data
            gaze_movement_df = pd.DataFrame(
                {
                    "Time": bins[:-1],
                    "Movement": movement_per_bin.values,
                    "deltaX": deltaX_per_bin.values,
                    "deltaY": deltaY_per_bin.values,
                }
            )

            print("Final gaze movement data columns:", gaze_movement_df.columns.tolist())
            print("Final gaze movement data sample:", gaze_movement_df.head())
            print("Final gaze movement data shape:", gaze_movement_df.shape)

            result["gaze_movement_data"] = gaze_movement_df
            result["max_time"] = max(result["max_time"], max_time)

            # Process events if present
            if "Event" in gaze_df.columns:
                event_df = gaze_df.dropna(subset=["Event", "Time"])
                if not event_df.empty:
                    result["event_data"] = event_df[["Time", "Event"]].copy()
                    result["max_time"] = max(result["max_time"], event_df["Time"].max())

        except Exception as e:
            logging.warning(f"Failed to load gaze file and process it: {e}")
            print(f"Error details: {str(e)}")

    return result


@st.cache_data(show_spinner=True)
def process_and_plot_data(
    name: str,
    eeg_df: pd.DataFrame | None,
    audio_io: io.BytesIO | None,
    gaze_df: pd.DataFrame | None,
    gaze_window_size: float = 0.1,  # in seconds
    cols: tuple[int, int] = (21, 25),
    **kwargs,
) -> tuple[plt.Figure, io.BytesIO | None, plt.Figure | None]:
    """
    Legacy function for backward compatibility.
    Plots EEG time series data and/or audio from a CSV file.
    """

    # Create a single plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define vertical offsets for different signals
    eeg_offset = 0
    audio_offset = -200

    # Define colors for different signals
    audio_color = "mediumslateblue"
    gaze_color = "red"

    if eeg_df is not None and len(eeg_df) > 1:
        mne_raw = preprocess_raw_data(eeg_df, cols, **kwargs)
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

    if audio_io is not None:
        try:
            # Load audio file
            y, sr = librosa.load(audio_io)
            audio_time = np.linspace(0, len(y) / sr, len(y))

            if eeg_df is not None:
                # Normalize audio to match EEG scale
                y_normalized = y * (np.max(eeg_df[eeg_columns].values) - np.min(eeg_df[eeg_columns].values)) / 2
                ax.plot(audio_time, y_normalized + audio_offset, color=audio_color, alpha=0.3, label="Audio")
            else:
                # Plot raw audio
                ax.plot(audio_time, y + audio_offset, color=audio_color, alpha=0.3, label="Audio")
        except Exception as e:
            logging.warning(f"Failed to load audio file and process it: {e}")

    # Plot gaze intensity if gaze file exists
    gaze_heatmap = None
    if gaze_df is not None and len(gaze_df) > 1:
        try:
            # --- Load gaze data ---
            gaze_heatmap = plot_gaze_heatmap(gaze_df)

            # Parse numeric X/Y and handle blinks ("." becomes NaN)
            gaze_df["X"] = pd.to_numeric(gaze_df["X"], errors="coerce")
            gaze_df["Y"] = pd.to_numeric(gaze_df["Y"], errors="coerce")

            # Parse TimeStamp and remove rows with invalid data
            gaze_df["TimeDT"] = pd.to_datetime(gaze_df["TimeStamp"], format="%H:%M:%S.%f", errors="coerce")
            gaze_df = gaze_df.dropna(subset=["TimeDT", "X", "Y"])  # Skip blinks

            # Create an explicit copy to avoid the SettingWithCopyWarning
            gaze_df = gaze_df.copy()

            # Compute elapsed time
            gaze_df.loc[:, "Time"] = (gaze_df["TimeDT"] - gaze_df["TimeDT"].min()).dt.total_seconds()

            # Compute delta movement (absolute change)
            gaze_df.loc[:, "deltaX"] = gaze_df["X"].diff().abs()
            gaze_df.loc[:, "deltaY"] = gaze_df["Y"].diff().abs()
            gaze_df.loc[:, "Movement"] = gaze_df["deltaX"] + gaze_df["deltaY"]

            # Ensure all values are positive
            gaze_df.loc[:, "deltaX"] = gaze_df["deltaX"].abs()
            gaze_df.loc[:, "deltaY"] = gaze_df["deltaY"].abs()
            gaze_df.loc[:, "Movement"] = gaze_df["Movement"].abs()

            # Bin into time windows
            window_size = gaze_window_size  # seconds
            max_time = gaze_df["Time"].max()
            bins = np.arange(0, max_time + window_size, window_size)
            gaze_df.loc[:, "bin"] = np.floor(gaze_df["Time"] / window_size).astype(int)

            # Sum movement per time window
            movement_per_bin = gaze_df.groupby("bin")["Movement"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)
            deltaX_per_bin = gaze_df.groupby("bin")["deltaX"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)
            deltaY_per_bin = gaze_df.groupby("bin")["deltaY"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)

            # Scale the summed values to 0-100 range
            max_movement = movement_per_bin.max()
            if max_movement > 0:
                movement_per_bin = (movement_per_bin / max_movement) * 100
                deltaX_per_bin = (deltaX_per_bin / deltaX_per_bin.max()) * 100
                deltaY_per_bin = (deltaY_per_bin / deltaY_per_bin.max()) * 100

            # Prepare gaze movement data
            gaze_movement_df = pd.DataFrame(
                {
                    "Time": bins[:-1],
                    "Movement": movement_per_bin.values,
                    "deltaX": deltaX_per_bin.values,
                    "deltaY": deltaY_per_bin.values,
                }
            )

            print("Final gaze movement data columns:", gaze_movement_df.columns.tolist())
            print("Final gaze movement data sample:", gaze_movement_df.head())
            print("Final gaze movement data shape:", gaze_movement_df.shape)

            result["gaze_movement_data"] = gaze_movement_df
            result["max_time"] = max(result["max_time"], max_time)

            # Process events if present
            if "Event" in gaze_df.columns:
                event_df = gaze_df.dropna(subset=["Event", "Time"])
                unique_events = event_df["Event"].unique()
                # Fix colormap reference - use discrete color values from tab10
                cmap = plt.cm.get_cmap("tab10")
                event_colors = [cmap(i) for i in range(10)]  # Get 10 colors from the tab10 colormap
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
    if eeg_df is not None:
        title_parts.append("EEG")
    if audio_io is not None:
        title_parts.append("Audio")
    if gaze_df is not None:
        title_parts.append("Gaze")

    title = " + ".join(title_parts) if title_parts else "No Data Selected"
    ax.set_title(f"{title} - {name}")

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

    return fig, audio_io, gaze_heatmap


def plot_interactive_eeg(
    data: pd.DataFrame, channels: list, x_min: float = 0.0, x_max: float | None = None
) -> alt.Chart:
    """Create an interactive EEG chart with Altair"""
    selection = alt.selection_point(fields=["Channel"], bind="legend")

    # Use provided x-axis range or calculate from data
    if x_max is None:
        x_max = data["Time"].max()

    # Create a scale indicator
    scale_text = (
        alt.Chart(pd.DataFrame({"x": [x_min + 1], "y": [0], "text": ["1s"]}))
        .mark_text(align="left", baseline="top", dx=5, dy=5)
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    base = alt.Chart(data).encode(
        x=alt.X("Time:Q", title="Time (s)", scale=alt.Scale(domain=[x_min, x_max])),
        y=alt.Y("Value:Q", title="Signal Amplitude"),
        color=alt.Color(
            "Channel:N",
            scale=alt.Scale(scheme="category10"),
            legend=alt.Legend(
                orient="top",  # Place legend at the top
                columns=3,  # Arrange in 3 columns
                symbolSize=100,  # Make symbols smaller
                labelFontSize=10,  # Make labels smaller
                title=None,  # Remove title
                padding=5,  # Add some padding
            ),
        ),
    )

    lines = (
        base.mark_line().encode(opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_params(selection)
    )

    return (lines + scale_text).properties(
        width=700,
        height=400,
        title="EEG Signals",
        padding={"left": 0, "right": 0, "top": 40, "bottom": 40},  # Increased padding for better alignment
        autosize=alt.AutoSizeParams(type="fit", contains="padding", resize=True),
    )


def plot_interactive_audio(
    data: pd.DataFrame, x_min: float = 0.0, x_max: float | None = None, downsample_factor: int = 20
) -> alt.Chart:
    """Create an interactive audio waveform chart with Altair

    Args:
        data: DataFrame containing Time and Value columns
        x_min: Minimum time value to display
        x_max: Maximum time value to display
        downsample_factor: Factor by which to reduce the number of data points (default: 10)
    """
    # Use provided x-axis range or calculate from data
    if x_max is None:
        x_max = data["Time"].max()

    # Downsample the data
    if downsample_factor > 1:
        data = data.iloc[::downsample_factor]

    # Create a scale indicator
    scale_text = (
        alt.Chart(pd.DataFrame({"x": [x_min + 1], "y": [0], "text": ["1s"]}))
        .mark_text(align="left", baseline="top", dx=5, dy=5)
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    chart = (
        alt.Chart(data)
        .mark_line(color="mediumslateblue", opacity=0.6)
        .encode(
            x=alt.X("Time:Q", title="Time (s)", scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y("Value:Q", title="Amplitude"),
        )
    )

    return (chart + scale_text).properties(width=700, height=200, title="Audio Waveform")


def plot_interactive_gaze_movement(data: pd.DataFrame, x_min: float = 0.0, x_max: float | None = None) -> alt.Chart:
    """Create an interactive gaze movement chart with Altair"""
    # Print data info for debugging
    print("Plot function - Data columns:", data.columns.tolist())
    print("Plot function - Data sample:", data.head())
    print("Plot function - Data shape:", data.shape)

    # Use provided x-axis range or calculate from data
    if x_max is None:
        x_max = data["Time"].max()

    # Create a scale indicator
    scale_text = (
        alt.Chart(pd.DataFrame({"x": [x_min + 1], "y": [0], "text": ["1s"]}))
        .mark_text(align="left", baseline="top", dx=5, dy=5)
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    # Create base chart with proper encoding
    base = alt.Chart(data).encode(x=alt.X("Time:Q", title="Time (s)", scale=alt.Scale(domain=[x_min, x_max])))

    # Create X movement area
    x_area = base.mark_area(color="red", opacity=0.5).encode(
        y=alt.Y("deltaX:Q", scale=alt.Scale(zero=False)), tooltip=["Time:Q", "deltaX:Q"]
    )

    # Create Y movement area
    y_area = base.mark_area(color="blue", opacity=0.5).encode(
        y=alt.Y("deltaY:Q", scale=alt.Scale(zero=False)), tooltip=["Time:Q", "deltaY:Q"]
    )

    # Combine the charts with proper layering
    chart = (x_area + y_area + scale_text).properties(title="Gaze Movement Intensity", height=200)

    return chart


def plot_interactive_events(
    data: pd.DataFrame | None, x_min: float = 0.0, x_max: float | None = None
) -> Optional[alt.Chart]:
    """Create an interactive events chart with Altair showing events as a single timeline"""
    if data is None or data.empty:
        return None

    # Add a constant column for the y-axis
    data = data.copy()
    data["y"] = 0

    # Use provided x-axis range or calculate from data
    if x_max is None:
        x_max = data["Time"].max()

    # Create a scale indicator
    scale_text = (
        alt.Chart(pd.DataFrame({"x": [x_min + 1], "y": [0], "text": ["1s"]}))
        .mark_text(align="left", baseline="top", dx=5, dy=5)
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    # Create a point chart for events on a single line
    chart = (
        alt.Chart(data)
        .mark_circle(size=80)
        .encode(
            x=alt.X("Time:Q", title="Time (s)", scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y("y:Q", title=None, axis=None),  # Fixed position on a single line
            color=alt.Color("Event:N", scale=alt.Scale(scheme="category10")),
            tooltip=["Time:Q", "Event:N"],
            opacity=alt.value(0.8),
            size=alt.Size("Event:N", legend=None, scale=alt.Scale(range=[50, 200])),
        )
    )

    # Add a rule to represent the timeline
    timeline = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")

    # Add text labels for events
    event_labels = chart.mark_text(align="center", baseline="bottom", dy=-10, fontSize=10).encode(text="Event:N")

    # Combine the visualizations
    combined = (timeline + chart + event_labels + scale_text).properties(width=700, height=100, title="Event Timeline")

    return combined
