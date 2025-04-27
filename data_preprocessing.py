import os
import re
import pandas as pd
import mne
import numpy as np


def parse_filename(filename):
    basename = os.path.basename(filename)
    basename, _ = os.path.splitext(basename)
    pattern = (
        r"P(?P<participant_id>\d+)-(?P<order>\d+)-S(?P<sentence_id>\d+)-"
        r"(?P<cond>[AU])-i(?P<cong>[CIM])(?P<image_id>\w+)"
    )
    m = re.match(pattern, basename)
    if m:
        return m.groupdict()
    else:
        return None


def load_csv_to_raw(csv_file, sfreq=256, drop_blink=True):
    df = pd.read_csv(csv_file)

    if drop_blink:
        df = df[df["Elements"].str.lower() != "blink"].reset_index(drop=True)

    eeg_band_channels = [
        "Delta_TP9",
        "Delta_AF7",
        "Delta_AF8",
        "Delta_TP10",
        "Theta_TP9",
        "Theta_AF7",
        "Theta_AF8",
        "Theta_TP10",
        "Alpha_TP9",
        "Alpha_AF7",
        "Alpha_AF8",
        "Alpha_TP10",
        "Beta_TP9",
        "Beta_AF7",
        "Beta_AF8",
        "Beta_TP10",
        "Gamma_TP9",
        "Gamma_AF7",
        "Gamma_AF8",
        "Gamma_TP10",
    ]

    raw_eeg_channels = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

    hsi_channels = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]

    sfreq = 256

    df.fillna(method="ffill", inplace=True)

    # Extract data for each group (transpose to get shape: n_channels x n_samples)
    data_eeg_band = df[eeg_band_channels].to_numpy().T
    data_raw_eeg = df[raw_eeg_channels].to_numpy().T
    data_hsi = df[hsi_channels].to_numpy().T

    # Create MNE Info objects for each group with appropriate channel types.
    # For EEG and spectral features, 'eeg' is appropriate. For others, you might use 'misc' or specific types.
    info_eeg_band = mne.create_info(
        ch_names=eeg_band_channels,
        sfreq=sfreq,
        ch_types=["eeg"] * len(eeg_band_channels),
    )
    info_raw_eeg = mne.create_info(ch_names=raw_eeg_channels, sfreq=sfreq, ch_types=["eeg"] * len(raw_eeg_channels))
    info_hsi = mne.create_info(ch_names=hsi_channels, sfreq=sfreq, ch_types=["misc"] * len(hsi_channels))

    # Create MNE Raw objects
    raw_eeg_band = mne.io.RawArray(data_eeg_band, info_eeg_band)
    raw_raw_eeg = mne.io.RawArray(data_raw_eeg, info_raw_eeg)
    raw_hsi = mne.io.RawArray(data_hsi, info_hsi)

    return raw_eeg_band, raw_raw_eeg, raw_hsi


def preprocess_raw_data(
    file,
    cols: tuple[int, int],
    *,
    bandpass: None | tuple[float, float] = (1.0, 50.0),
    notch_filter: None | int = 50,
    ica: bool = True,
    out_file=None,
) -> mne.io.RawArray | None:
    raw_eeg_channels = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
    df = pd.read_csv(file)

    if df.empty:
        return None

    raw_eeg_channels = list(df.columns[cols[0] : cols[1]])

    # Convert TimeStamp from HH:MM:SS.sss format to seconds
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="%H:%M:%S.%f", errors="coerce")
    df["TimeStamp"] = (df["TimeStamp"] - df["TimeStamp"].iloc[0]).dt.total_seconds()

    # Forward-fill missing data
    df.ffill(inplace=True)
    # Ensure no NaN values remain by filling any remaining ones with 0
    df.fillna(0, inplace=True)

    # Create MNE RawArray (you’ll need to reshape and adjust metadata accordingly)
    sfreq = 256  # MUSE2 typically samples at 256Hz
    info = mne.create_info(ch_names=raw_eeg_channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(df[raw_eeg_channels].T.values, info)

    if bandpass is not None:
        raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

    if notch_filter is not None:
        raw.notch_filter(freqs=notch_filter)

    if ica:
        raw = apply_ica(raw, len(raw_eeg_channels))

    if out_file is not None:
        clean_data = raw.get_data()
        np.savetxt(out_file, clean_data.T, delimiter=",")

    return raw


def apply_ica(
    raw: mne.io.RawArray, num_components, *, method: str = "fastica", auto_exclude=True, plot=False
) -> mne.io.RawArray:
    # MNE Montage expects names without the RAW_ part
    rename_dict = {
        "RAW_TP9": "TP9",
        "RAW_AF7": "AF7",
        "RAW_AF8": "AF8",
        "RAW_TP10": "TP10",
    }

    # Keep all other channels unchanged
    raw.rename_channels({ch: rename_dict.get(ch, ch) for ch in raw.ch_names})

    # Assign electrode positions
    montage = mne.channels.make_standard_montage("standard_1020")

    # Set montage with on_missing parameter to ignore missing channels
    raw.set_montage(montage, on_missing="ignore")

    ica = mne.preprocessing.ICA(n_components=num_components, random_state=42, method=method)
    ica.fit(raw)

    if plot:
        ica.plot_components()
        ica.plot_sources(raw)

    if auto_exclude:
        # Extract ICA source time series
        sources = ica.get_sources(raw).get_data()

        # Heuristic 1: high peak-to-peak amplitude
        ptp_amplitudes = np.ptp(sources, axis=1)
        ptp_threshold = np.percentile(ptp_amplitudes, 90)  # top 5% most extreme
        bad_ptp = np.where(ptp_amplitudes > ptp_threshold)[0]

        # Heuristic 2: high kurtosis
        from scipy.stats import kurtosis

        kurt = kurtosis(sources, axis=1)
        kurt_threshold = np.percentile(kurt, 90)
        bad_kurt = np.where(kurt > kurt_threshold)[0]

        # Union of detected bad components
        bad_components = set(bad_ptp).union(bad_kurt)

        ica.exclude = list(bad_components)
        print(f"Automatically excluding ICA components: {ica.exclude}")

    ica.apply(raw)

    if plot:
        raw.plot()

    return raw


if __name__ == "__main__":
    preprocess_raw_data("./ufal_emmt/preprocessed-data/eeg/See/P43-32-S191-A-U.csv", ica=True)
