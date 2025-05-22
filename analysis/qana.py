# ------------------------------------------------------------
#  AUDIO / GAZE helpers  (unchanged API, new **relative** time axis)
# ------------------------------------------------------------
import io, librosa, numpy as np, pandas as pd
from typing import Any, List, Dict


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


# ------------------------------------------------------------
#  _gaze_to_dataframe  — add empty-frame guard
# ------------------------------------------------------------
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


# ------------------------------------------------------------
#  Correlation helper – only _prep() changed
# ------------------------------------------------------------
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


import sys

sys.path.append("../")

import matplotlib.pyplot as plt
from src.file_loader import FileLoader
import pandas as pd
import numpy as np


color_map = {
    "RAW_TP9": "tab:blue",
    "RAW_AF7": "tab:orange",
    "RAW_AF8": "tab:green",
    "RAW_TP10": "tab:red",
}
gray_shade = {
    "RAW_TP9": "gray",
    "RAW_AF7": "dimgray",
    "RAW_AF8": "darkgray",
    "RAW_TP10": "slategray",
}


# ---------------------------------------------------------------------
def _mad(x: pd.Series) -> float:
    """Median absolute deviation scaled to σ."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


# ---------------------------------------------------------------------
def is_eeg_record_valid(
    eeg_df: pd.DataFrame,
    *,
    # ----------- thresholds you may want to tune ------------------
    flat_mad_thr: float = 2.5,  #  ≲  2.5  Muse-raw units ⇒ flat
    wild_mad_thr: float = 120,  #  ≳ 120 units ⇒ wild
    wild_jump_thr: float = 150,  # 95-th pct(|Δ|) ≳ 150 ⇒ wild
) -> dict:
    """
    Returns
    -------
    {channel_name: (is_valid: bool, explanation: str)}
    """
    chans = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
    sig = eeg_df[chans].astype(float)

    results = {}
    for ch in chans:
        s = sig[ch].dropna()
        mad = _mad(s)
        p95_jump = s.diff().abs().quantile(0.95)

        # ---------------- failure modes ----------------------------
        if mad < flat_mad_thr:
            results[ch] = (False, f"MAD {mad:.1f} < {flat_mad_thr})")
        elif mad > wild_mad_thr or p95_jump > wild_jump_thr:
            results[ch] = (
                False,
                f"MAD {mad:.0f} > {wild_mad_thr} or " f"95-pct|Δ| {p95_jump:.0f} > {wild_jump_thr})",
            )
        else:
            results[ch] = (True, "normal")

    return results


def data_generator():
    loader = FileLoader(base_path="../ufal_emmt")
    data_units = loader.get_data_files(
        category_filter="Read", participant_id_filter=None, sentence_id_filter=None, sort_by="participant_id"
    )

    print(data_units[0])

    for read_data_unit in data_units:
        participant_id = read_data_unit["participant_id"]
        sentence_id = read_data_unit["sentence_id"]

        translate_data_units = loader.get_data_files(
            category_filter="Translate", participant_id_filter=participant_id, sentence_id_filter=sentence_id
        )
        see_data_units = loader.get_data_files(
            category_filter="See", participant_id_filter=participant_id, sentence_id_filter=sentence_id
        )
        update_data_units = loader.get_data_files(
            category_filter="Update", participant_id_filter=participant_id, sentence_id_filter=sentence_id
        )

        rd = loader.load_data(read_data_unit)
        td = loader.load_data(translate_data_units[0])
        sd = loader.load_data(see_data_units[0])
        ud = loader.load_data(update_data_units[0])

        yield (rd, td, sd, ud), read_data_unit


# ------------------------------------------------------------
# 1)  Helper that yields the 32 four-tuples for one participant
# ------------------------------------------------------------
def participant_sequences(loader: FileLoader, participant_id: str):
    """
    Yields exactly 32 tuples (rd, td, sd, ud) for the requested participant,
    ordered by the `order` field (1‒32).

    Parameters
    ----------
    loader : FileLoader
    participant_id : str
    """
    read_units = loader.get_data_files(
        category_filter="Read",
        participant_id_filter=participant_id,
        sentence_id_filter=None,
    )

    # Manually sort by order field
    read_units = sorted(read_units, key=lambda x: int(x["order"]))

    if len(read_units) != 32:
        raise ValueError(f"Expected 32 Read units, found {len(read_units)} for participant {participant_id}")

    for read_du in read_units:  # already sorted by order
        sentence_id = read_du["sentence_id"]

        # One data-unit per category & sentence
        translate_dus = loader.get_data_files("Translate", participant_id, sentence_id)
        see_dus = loader.get_data_files("See", participant_id, sentence_id)
        update_dus = loader.get_data_files("Update", participant_id, sentence_id)

        if len(translate_dus) == 1 and len(see_dus) == 1 and len(update_dus) == 1:
            translate_du = translate_dus[0]
            see_du = see_dus[0]
            update_du = update_dus[0]
        else:
            missing = "translate " if len(translate_dus) == 0 else ""
            missing += "see " if len(see_dus) == 0 else ""
            missing += "update" if len(update_dus) == 0 else ""
            print(f"Skipping participant {participant_id} and sentence {sentence_id}, missing parts: {missing}")
            continue

        yield (
            loader.load_data(read_du),
            loader.load_data(translate_du),
            loader.load_data(see_du),
            loader.load_data(update_du),
            read_du["order"],  # keep the order number for sorting/plotting
        )


# ------------------------------------------------------------
# 2)  Plot all 32 sequences for one participant in one figure
# ------------------------------------------------------------
def plot_participant_eeg(participant_id: str, base_path: str = "../ufal_emmt"):
    """
    Creates a 32×4 figure:
        • rows  = order 1‒32
        • cols  = Read | Translate | See | Update
    """
    loader = FileLoader(base_path=base_path)
    seq_iter = sorted(participant_sequences(loader, participant_id), key=lambda x: x[-1])  # just in case

    fig, axes = plt.subplots(32, 4, figsize=(20, 2 * 32 + 2), sharey="row", constrained_layout=True)
    plt.subplots_adjust(hspace=0.6)
    col_labels = ["Read", "Translate", "See", "Update"]

    for row_idx, (rd, td, sd, ud, order) in enumerate(seq_iter):
        for col_idx, (eeg_df, label) in enumerate(zip([rd[0], td[0], sd[0], ud[0]], col_labels)):
            ax = axes[row_idx, col_idx]

            # validity & colour selection (as in your code)
            results = is_eeg_record_valid(eeg_df)
            eeg_df["TimeStamp"] = pd.to_datetime(eeg_df["TimeStamp"], format="%H:%M:%S.%f")

            for ch, ch_label in zip(
                ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"],
                ["TP9", "AF7", "AF8", "TP10"],
            ):
                ok, reason = results[ch]
                colour = color_map[ch] if ok else gray_shade[ch]
                ax.plot(eeg_df["TimeStamp"], eeg_df[ch], colour, label=ch_label)

            # only add legend on first row to keep the figure tidy
            if row_idx == 0:
                ax.set_title(label)
            if col_idx == 0:
                ax.set_ylabel(f"order {order}")
            if row_idx == 31:  # last row
                ax.set_xlabel("Time")
            ax.grid(True)

    # put one common legend outside the grid
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),  # exactly at the top edge
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"EEG – participant {participant_id}",
        y=0.97,  # inside the figure now
        fontsize=18,
    )

    plt.show()
    return fig


if __name__ == "__main__":
    g = data_generator()
    n = 0
    for (rd, td, sd, ud), du in g:
        if n > 100:
            break
        n += 1

        r_eeg_df, r_gaze_df, r_audio_io = rd
        t_eeg_df, t_gaze_df, t_audio_io = td
        s_eeg_df, s_gaze_df, s_audio_io = sd
        u_eeg_df, u_gaze_df, u_audio_io = ud

        r_corr = compute_correlations(*rd)
        t_corr = compute_correlations(*td)
        s_corr = compute_correlations(*sd)
        u_corr = compute_correlations(*ud)

        print(f"Read correlations  : {r_corr}")
        print(f"Translate corr.    : {t_corr}")
        print(f"See correlations   : {s_corr}")
        print(f"Update correlations: {u_corr}")

        fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
        plt.tight_layout(rect=(0, 0, 1, 0.95))  # Make room for the suptitle
        plt.suptitle(
            f"EEG | participant_id: {du['participant_id']} | sentence_id: {du['sentence_id']} | order: {du['order']}",
            fontsize=16,
        )

        for i, (eeg_df, label, ax) in enumerate(
            zip([r_eeg_df, t_eeg_df, s_eeg_df, u_eeg_df], ["Read", "Translate", "See", "Update"], axes)
        ):
            # Compute validity
            results = is_eeg_record_valid(eeg_df)
            overall_valid = all(ok for ok, _ in results.values())

            # Convert TimeStamp
            eeg_df["TimeStamp"] = pd.to_datetime(eeg_df["TimeStamp"], format="%H:%M:%S.%f")

            # Plot each channel
            for ch, ch_label in zip(["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"], ["TP9", "AF7", "AF8", "TP10"]):
                ok, reason = results[ch]
                color = color_map[ch] if ok else gray_shade[ch]
                ax.plot(eeg_df["TimeStamp"], eeg_df[ch], label=ch_label, color=color)
                print(f"{label} - {ch:7s} -> {'OK' if ok else 'BAD'} | {reason}")

            ax.set_title(f"{label} EEG\nValid: {overall_valid}")
            ax.set_xlabel("Time")
            ax.grid(True)
            if i == 0:
                ax.set_ylabel("Amplitude")
            ax.legend()

        plt.tight_layout()
        plt.show()
