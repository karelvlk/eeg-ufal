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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, List, Dict, Literal
from src.data_preprocessing import preprocess_raw_data


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
    Plot EEG time series data and/or audio from a CSV file.
    Args:
        csv_file: Path to the CSV file
        cols: Tuple of column indices to use
        plot_eeg: Whether to plot EEG data
        plot_audio: Whether to plot audio data
        plot_gaze: Whether to plot gaze data
        gaze_window_size: Size of the time window for gaze intensity sampling in seconds
        **kwargs: Additional arguments passed to get_data
    Returns:
        Figure for Streamlit to display
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

            # Bin into time windows
            window_size = gaze_window_size  # seconds
            max_time = gaze_df["Time"].max()
            bins = np.arange(0, max_time + window_size, window_size)
            gaze_df.loc[:, "bin"] = np.floor(gaze_df["Time"] / window_size).astype(int)

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


def create_interactive_plot(
    name: str,
    eeg_df: pd.DataFrame | None,
    audio_io: io.BytesIO | None,
    gaze_df: pd.DataFrame | None,
    gaze_window_size: float = 0.1,  # in seconds
    cols: tuple[int, int] = (21, 25),
    **kwargs,
) -> tuple[go.Figure, go.Figure | None]:
    """
    Create an interactive Plotly figure combining EEG, audio, and gaze data.
    Args:
        name: Name of the data unit
        eeg_df: DataFrame containing EEG data
        audio_io: BytesIO object containing audio data
        gaze_df: DataFrame containing gaze data
        gaze_window_size: Size of the time window for gaze intensity sampling in seconds
        cols: Tuple of column indices to use for EEG data
        **kwargs: Additional arguments passed to preprocess_raw_data
    Returns:
        Tuple of Plotly Figure objects (main plot and heatmap)
    """
    # Create main figure with 4 subplots
    main_fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("EEG Signals", "Audio", "Gaze Intensity (ΔX positive, ΔY negative)", "Events"),
        row_heights=[0.5, 0.2, 0.2, 0.1],
    )

    # Set main figure height to 900px
    main_fig.update_layout(
        height=900,
        title_text=name,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
        margin=dict(t=50, b=20),
    )

    # Add vertical line shapes for all subplots
    for row in range(1, 5):
        main_fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            row=row,
            col=1,
            visible=True,  # Initially hidden
            name=f"vline_{row}",
        )

    # Add hover functionality to show vertical lines
    main_fig.update_layout(
        hovermode="x unified",
        hoverdistance=100,
        spikedistance=1000,
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="gray",
            spikethickness=1,
            spikedash="dash",
            showgrid=True,
        ),
        xaxis2=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="gray",
            spikethickness=1,
            spikedash="dash",
            showgrid=True,
        ),
        xaxis3=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="gray",
            spikethickness=1,
            spikedash="dash",
            showgrid=True,
        ),
        xaxis4=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="gray",
            spikethickness=1,
            spikedash="dash",
            showgrid=True,
        ),
        yaxis=dict(
            showgrid=True,
        ),
        yaxis2=dict(
            showgrid=True,
        ),
        yaxis3=dict(
            showgrid=True,
        ),
        yaxis4=dict(
            showgrid=True,
        ),
    )

    # Initialize variables for data range calculation
    eeg_min = 0
    eeg_max = 0
    eeg_range = 0

    if eeg_df is not None and len(eeg_df) > 1:
        mne_raw = preprocess_raw_data(eeg_df, cols, **kwargs)
        if mne_raw is not None:
            # Convert MNE Raw data back to Pandas DataFrame
            data = mne_raw.get_data()  # Get the EEG data
            time = mne_raw.times  # Get the time vector
            eeg_columns = mne_raw.ch_names  # Get the channel names

            # Calculate EEG data range for normalization
            eeg_min = np.min(data)
            eeg_max = np.max(data)
            eeg_range = eeg_max - eeg_min

            # Plot EEG data
            for col in eeg_columns:
                main_fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=data[eeg_columns.index(col)],
                        name=col,
                        mode="lines",
                        visible="legendonly" if len(eeg_columns) > 5 else True,
                    ),
                    row=1,
                    col=1,
                )

    # Define colors for different signals
    audio_color = "mediumslateblue"
    gaze_color = "red"

    if audio_io is not None:
        try:
            # Load audio file
            y, sr = librosa.load(audio_io)
            audio_time = np.linspace(0, len(y) / sr, len(y))

            # Normalize audio to its own scale
            y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

            main_fig.add_trace(
                go.Scatter(
                    x=audio_time,
                    y=y_normalized,
                    name="Audio",
                    mode="lines",
                    line=dict(color=audio_color),
                    opacity=0.3,
                ),
                row=2,
                col=1,
            )
        except Exception as e:
            logging.warning(f"Failed to load audio file and process it: {e}")

    # Plot gaze intensity if gaze file exists
    if gaze_df is not None and len(gaze_df) > 1:
        try:
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

            # Bin into time windows
            window_size = gaze_window_size  # seconds
            max_time = gaze_df["Time"].max()
            bins = np.arange(0, max_time + window_size, window_size)
            gaze_df.loc[:, "bin"] = np.floor(gaze_df["Time"] / window_size).astype(int)

            # Sum movement per time window for X and Y separately
            deltaX_per_bin = gaze_df.groupby("bin")["deltaX"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)
            deltaY_per_bin = gaze_df.groupby("bin")["deltaY"].sum().reindex(np.arange(len(bins) - 1), fill_value=0)

            # Normalize to 0-1 range
            max_delta = max(deltaX_per_bin.max(), deltaY_per_bin.max())
            deltaX_normalized = deltaX_per_bin / max_delta
            deltaY_normalized = -deltaY_per_bin / max_delta  # Negative for Y

            # Add deltaX bars (positive)
            main_fig.add_trace(
                go.Bar(
                    x=bins[:-1],
                    y=deltaX_normalized,
                    name="ΔX (Horizontal)",
                    marker_color="rgba(255, 99, 71, 0.7)",  # Tomato color with 70% opacity
                    opacity=0.7,
                    width=window_size * 0.8,  # Slightly narrower bars
                    hovertemplate="Time: %{x:.2f}s<br>ΔX: %{y:.2f}<extra></extra>",
                ),
                row=3,
                col=1,
            )

            # Add deltaY bars (negative)
            main_fig.add_trace(
                go.Bar(
                    x=bins[:-1],
                    y=deltaY_normalized,
                    name="ΔY (Vertical)",
                    marker_color="rgba(65, 105, 225, 0.7)",  # Royal blue color with 70% opacity
                    opacity=0.7,
                    width=window_size * 0.8,  # Slightly narrower bars
                    hovertemplate="Time: %{x:.2f}s<br>ΔY: %{y:.2f}<extra></extra>",
                ),
                row=3,
                col=1,
            )

            # Add a zero line for reference
            main_fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                line_width=1,
                row=3,
                col=1,
            )

            # Add event markers if present
            if "Event" in gaze_df.columns:
                event_df = gaze_df.dropna(subset=["Event", "Time"])
                unique_events = event_df["Event"].unique()

                # Define a color palette for events
                event_colors = {
                    "saccade": "rgba(255, 140, 0, 0.8)",  # Dark orange
                    "fixation": "rgba(0, 128, 0, 0.8)",  # Dark green
                    "blink": "rgba(128, 0, 128, 0.8)",  # Purple
                    "default": "rgba(0, 0, 255, 0.8)",  # Blue for other events
                }

                for event in unique_events:
                    times = event_df.loc[event_df["Event"] == event, "Time"].values
                    color = event_colors.get(event.lower(), event_colors["default"])

                    main_fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=[0.5] * len(times),  # Center in the event plot
                            mode="markers",
                            name=f"Event: {event}",
                            marker=dict(
                                symbol="line-ns",
                                size=12,  # Slightly larger
                                line=dict(width=2, color=color),
                            ),
                            hovertemplate="Time: %{x:.2f}s<br>Event: %{text}<extra></extra>",
                            text=[event] * len(times),  # Show event type in hover
                        ),
                        row=4,
                        col=1,
                    )

        except Exception as e:
            logging.warning(f"Failed to load gaze file and process it: {e}")

    # Create heatmap figure if gaze data exists
    heatmap_fig = None
    if gaze_df is not None and len(gaze_df) > 1:
        try:
            # Ensure X and Y are numeric
            gaze_df["X"] = pd.to_numeric(gaze_df["X"], errors="coerce")
            gaze_df["Y"] = pd.to_numeric(gaze_df["Y"], errors="coerce")
            # Drop rows with invalid coordinates
            gaze_df = gaze_df.dropna(subset=["X", "Y"])

            # Split data by event type
            saccade_df = gaze_df[gaze_df["Event"] == "saccade"]
            fixation_df = gaze_df[gaze_df["Event"] == "fixation"]

            # Create heatmap figure with 3 subplots
            heatmap_fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=("Saccade KDE", "Fixation KDE", "Combined KDE"),
                horizontal_spacing=0.05,
            )

            # Define common axis ranges
            x_min, x_max = 0, 1280
            y_min, y_max = 0, 1024

            # Create grid for KDE evaluation
            x_grid = np.linspace(x_min, x_max, 100)
            y_grid = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([X.ravel(), Y.ravel()])

            def create_kde_plot(x, y, colorscale, name):
                # Calculate KDE using scipy
                from scipy.stats import gaussian_kde

                values = np.vstack([x, y])
                kernel = gaussian_kde(values)
                Z = np.reshape(kernel(positions).T, X.shape)

                return go.Heatmap(
                    z=Z,
                    x=x_grid,
                    y=y_grid,
                    colorscale=colorscale,
                    name=name,
                    showscale=True,
                    hoverongaps=False,
                )

            # Add saccade KDE
            if len(saccade_df) > 0:
                heatmap_fig.add_trace(
                    create_kde_plot(saccade_df["X"], saccade_df["Y"], "Reds", "Saccade"),
                    row=1,
                    col=1,
                )

            # Add fixation KDE
            if len(fixation_df) > 0:
                heatmap_fig.add_trace(
                    create_kde_plot(fixation_df["X"], fixation_df["Y"], "Blues", "Fixation"),
                    row=1,
                    col=2,
                )

            # Add combined KDE
            heatmap_fig.add_trace(
                create_kde_plot(gaze_df["X"], gaze_df["Y"], "Viridis", "Combined"),
                row=1,
                col=3,
            )

            # Update layout for heatmaps
            heatmap_fig.update_layout(
                title="Gaze KDE Heatmaps",
                height=400,  # Fixed height for heatmaps
                showlegend=False,
                margin=dict(t=50, b=20),
            )

            # Update axes for all heatmaps
            for col in range(1, 4):
                heatmap_fig.update_xaxes(
                    title_text="X Position",
                    range=[x_min, x_max],
                    row=1,
                    col=col,
                )
                heatmap_fig.update_yaxes(
                    title_text="Y Position",
                    range=[y_max, y_min],  # Inverted y-axis to put (0,0) at top-left
                    row=1,
                    col=col,
                )

        except Exception as e:
            logging.warning(f"Failed to create gaze heatmap: {e}")

    return main_fig, heatmap_fig


# ------------------------------------------------------------
# Correlation analysis functions from qana.py
# ------------------------------------------------------------
def _audio_to_dataframe(audio_like: Any, *, target_sr: int | None = 1000, col_name: str = "Amplitude") -> pd.DataFrame:
    if isinstance(audio_like, pd.DataFrame):
        data_cols = audio_like.columns.difference(["TimeStamp"]).tolist()
        return audio_like.rename(columns={data_cols[0]: col_name}).copy()

    if isinstance(audio_like, tuple) and len(audio_like) == 2:
        samples, sr = audio_like
    else:
        if isinstance(audio_like, (io.IOBase, io.BytesIO)):
            audio_like.seek(0)
        samples, sr = librosa.load(audio_like, sr=None, mono=True)

    if target_sr and sr != target_sr:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # ----- relative timeline (Timedelta index) ---------------------------
    ts = pd.to_timedelta(np.arange(len(samples)) / sr, unit="s")
    return pd.DataFrame({"TimeStamp": ts, col_name: samples})


def _gaze_to_dataframe(
    gaze_df: pd.DataFrame, *, window_size: float | None = None, intensity_col: str = "Gaze_Intensity"
) -> pd.DataFrame:
    g = gaze_df.copy()
    g["X"] = pd.to_numeric(g["X"], errors="coerce")
    g["Y"] = pd.to_numeric(g["Y"], errors="coerce")
    g["TimeDT"] = pd.to_datetime(g["TimeStamp"], format="%H:%M:%S.%f", errors="coerce")
    g = g.dropna(subset=["TimeDT", "X", "Y"]).sort_values("TimeDT")

    # ---------- NEW: if nothing left, return empty shell -----------------
    if g.empty:
        return pd.DataFrame(columns=["TimeStamp", intensity_col])

    # ---------- movement & intensity ------------------------------------
    g["dX"] = g["X"].diff().abs()
    g["dY"] = g["Y"].diff().abs()
    g["intensity"] = g["dX"] + g["dY"]

    if window_size:
        g["elapsed"] = (g["TimeDT"] - g["TimeDT"].iloc[0]).dt.total_seconds()
        g["bin"] = np.floor(g["elapsed"] / window_size).astype(int)
        grp = g.groupby("bin")["intensity"].sum()
        max_delta = grp.max() or 1
        grp = grp / max_delta
        ts = pd.to_timedelta(grp.index * window_size, unit="s")
        return pd.DataFrame({"TimeStamp": ts, intensity_col: grp.values})

    ts = g["TimeDT"] - g["TimeDT"].iloc[0]  # relative timeline
    return pd.DataFrame({"TimeStamp": ts, intensity_col: g["intensity"].values})


def compute_correlations(
    eeg_df: pd.DataFrame,
    raw_gaze_df: pd.DataFrame,
    audio_like: Any,
    *,
    eeg_chans: List[str] | None = None,
    gaze_window_size: float | None = None,
    gaze_col: str | None = None,
    resample_ms: int | None = 100,
    method: str = "pearson",
) -> Dict[str, float]:
    if eeg_chans is None:
        eeg_chans = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

    if gaze_col is None:
        gaze_df = _gaze_to_dataframe(raw_gaze_df, window_size=gaze_window_size)
        gaze_col = "Gaze_Intensity"
    else:
        gaze_df = raw_gaze_df

    # ---------------- relative index helper ------------------------------
    def _prep(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        # empty guard
        if df.empty:
            return pd.DataFrame(columns=cols)

        d = df.copy()
        ts = d["TimeStamp"]

        # if not already a TimedeltaIndex, convert and make relative
        if not pd.api.types.is_timedelta64_dtype(ts.dtype):
            ts = pd.to_datetime(ts)  # parse wall‐clock times
            ts = ts - ts.iloc[0]  # make relative
        # else: ts is already a TimedeltaIndex

        d = d.set_index(ts)[cols]

        if resample_ms is not None:
            d = d.resample(f"{resample_ms}ms").mean()

        return d

    eeg_r = _prep(eeg_df, eeg_chans)
    audio_r = _prep(_audio_to_dataframe(audio_like, col_name="AUDIO"), ["AUDIO"])
    gaze_r = _prep(gaze_df, [gaze_col]).rename(columns={gaze_col: "GAZE"})

    merged = pd.concat([eeg_r, audio_r, gaze_r], axis=1).dropna()
    merged["EEG_mean"] = merged[eeg_chans].mean(axis=1)

    corr = {
        "4ch_audio": float(merged["EEG_mean"].corr(merged["AUDIO"], method=method)),
        "4ch_gaze": float(merged["EEG_mean"].corr(merged["GAZE"], method=method)),
    }
    for ch in eeg_chans:
        corr[f"{ch}_audio"] = float(merged[ch].corr(merged["AUDIO"], method=method))
        corr[f"{ch}_gaze"] = float(merged[ch].corr(merged["GAZE"], method=method))

    return corr


def create_correlation_plot(corr_dict: Dict[str, float]) -> go.Figure:
    """Create a bar chart visualization of the correlation values"""
    # Separate audio and gaze correlations
    audio_corrs = {k: v for k, v in corr_dict.items() if k.endswith("_audio")}
    gaze_corrs = {k: v for k, v in corr_dict.items() if k.endswith("_gaze")}

    # Create a figure with two subplots
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("EEG - Audio Correlations", "EEG - Gaze Correlations"), vertical_spacing=0.2
    )

    # Add audio correlation bars
    fig.add_trace(
        go.Bar(
            x=list(audio_corrs.keys()),
            y=list(audio_corrs.values()),
            name="Audio Correlation",
            marker_color="mediumslateblue",
        ),
        row=1,
        col=1,
    )

    # Add gaze correlation bars
    fig.add_trace(
        go.Bar(x=list(gaze_corrs.keys()), y=list(gaze_corrs.values()), name="Gaze Correlation", marker_color="tomato"),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="EEG Correlations with Audio and Gaze",
    )

    # Add y-axis range from -1 to 1 for correlation values
    fig.update_yaxes(range=[-1, 1], title_text="Correlation (Pearson)", row=1, col=1)
    fig.update_yaxes(range=[-1, 1], title_text="Correlation (Pearson)", row=2, col=1)

    # Add a horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Clean up x-axis labels
    fig.update_xaxes(
        title_text="",
        row=2,
        col=1,
        tickangle=45,
        tickmode="array",
        tickvals=list(gaze_corrs.keys()),
        ticktext=[k.replace("_gaze", "") for k in gaze_corrs.keys()],
    )

    fig.update_xaxes(
        title_text="",
        row=1,
        col=1,
        tickangle=45,
        tickmode="array",
        tickvals=list(audio_corrs.keys()),
        ticktext=[k.replace("_audio", "") for k in audio_corrs.keys()],
    )

    return fig
