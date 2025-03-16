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
    info_raw_eeg = mne.create_info(
        ch_names=raw_eeg_channels, sfreq=sfreq, ch_types=["eeg"] * len(raw_eeg_channels)
    )
    info_hsi = mne.create_info(
        ch_names=hsi_channels, sfreq=sfreq, ch_types=["misc"] * len(hsi_channels)
    )

    # Create MNE Raw objects
    raw_eeg_band = mne.io.RawArray(data_eeg_band, info_eeg_band)
    raw_raw_eeg = mne.io.RawArray(data_raw_eeg, info_raw_eeg)
    raw_hsi = mne.io.RawArray(data_hsi, info_hsi)

    return raw_eeg_band, raw_raw_eeg, raw_hsi
