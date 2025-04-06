import pandas as pd
import matplotlib.pyplot as plt


def process_and_plot_eeg_data(
    csv_file: str, cols: tuple[int, int] = (21, 25), **kwargs
) -> plt.Figure:
    """
    Plot EEG time series data from a CSV file.
    Returns the figure for Streamlit to display.
    """
    # Read the CSV
    df = pd.read_csv(csv_file)

    # Extract time series
    time = df["TimeStamp"]

    if isinstance(cols, tuple):
        if isinstance(cols[0], int):
            eeg_columns = df.columns[slice(*cols)]
        else:
            eeg_columns = list(cols)
    else:
        eeg_columns = [df.columns[cols]]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in eeg_columns:
        ax.plot(time, df[col], label=col)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EEG Signal Value")
    ax.set_title(f"EEG Time Series - {csv_file.name}")
    ax.legend()
    ax.grid(True)

    return fig
