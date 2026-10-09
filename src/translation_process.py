"""Human translation-process inspection. No inferred AOIs or semantic gold labels."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.dataset_cache import resolve_ui
from src.process_reviews import (diff_html, trial_id, LABELS, CONFIDENCES, save_review,
                                 read_reviews, latest_reviews, export_table, safe_csv, evidence_html)


def image_gaze_points(runs, rectangle, image_size):
    """Map screen fixation centers to image pixels; never rescale to gaze bounds."""
    left, top, width, height = map(float, rectangle)
    if not np.isfinite([left, top, width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Image rectangle must have finite coordinates and positive dimensions.")
    points = runs.copy()
    if points.empty:
        return points
    inside = (points.X.ge(left) & points.X.lt(left + width)
              & points.Y.ge(top) & points.Y.lt(top + height))
    points = points[inside].copy()
    points["image_x"] = (points.X - left) * image_size[0] / width
    points["image_y"] = (points.Y - top) * image_size[1] / height
    return points


def fixation_heatmap(points, image_size, smoothing=20.0):
    """Duration-weighted density in image pixels; relative intensity, not probability."""
    from scipy.ndimage import gaussian_filter
    width, height = image_size
    nx, ny = max(1, int(np.ceil(width / 5))), max(1, int(np.ceil(height / 5)))
    xedges, yedges = np.linspace(0, width, nx + 1), np.linspace(0, height, ny + 1)
    density = np.zeros((ny, nx))
    if not points.empty:
        density, _, _ = np.histogram2d(points.image_y, points.image_x,
                                       bins=(yedges, xedges), weights=points.duration_seconds)
        density = gaussian_filter(density, sigma=(smoothing / (height / ny), smoothing / (width / nx)), mode="constant")
        if density.max() > 0:
            density /= density.max()
    return (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2, density


def image_gaze_figure(image, points, show_points=False):
    import plotly.graph_objects as go
    width, height = image.size
    figure = go.Figure()
    figure.add_layout_image(source=image, x=0, y=0, sizex=width, sizey=height,
                            xref="x", yref="y", xanchor="left", yanchor="top",
                            sizing="stretch", layer="below")
    x, y, density = fixation_heatmap(points, image.size)
    figure.add_trace(go.Heatmap(
        x=x, y=y, z=density, zmin=0, zmax=1, zsmooth="best",
        colorscale=[[0, "rgba(0,0,0,0)"], [.08, "rgba(30,110,240,0.05)"],
                    [.25, "rgba(0,210,220,0.35)"], [.5, "rgba(255,230,50,0.55)"],
                    [.75, "rgba(255,110,20,0.70)"], [1, "rgba(230,30,30,0.80)"]],
        colorbar=dict(title="Relative<br>density", thickness=14, len=.85), showscale=bool(density.max()),
        hovertemplate="Relative density: %{z:.2f}<extra></extra>"))
    if show_points:
        figure.add_trace(go.Scatter(
        x=points.get("image_x", []), y=points.get("image_y", []), mode="markers",
        marker=dict(size=12, color=points.get("start_seconds", []), colorscale="Viridis",
                    opacity=.8, line=dict(width=1, color="white"),
                    colorbar=dict(title="Time (s)"), showscale=not points.empty),
        customdata=points[["start_seconds", "duration_seconds"]].to_numpy() if not points.empty else None,
        hovertemplate="Time: %{customdata[0]:.2f} s<br>Fixation: %{customdata[1]:.3f} s<extra></extra>"))
    figure.update_xaxes(range=[0, width], visible=False, showticklabels=False,
                        showgrid=False, zeroline=False, constrain="domain")
    figure.update_yaxes(range=[height, 0], visible=False, showticklabels=False,
                        showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    figure.update_layout(height=max(240, min(600, int(420 * height / width) + 40)),
                         margin=dict(l=0, r=0, t=0, b=0), dragmode="pan")
    return figure


def render_image_gaze(image_path, runs, phase, key, nested=False):
    import streamlit as st
    from PIL import Image
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    if phase not in ("See", "Update"):
        st.image(image, width=450)
        st.info("The assigned image was not shown during this phase; no photo overlay is displayed.")
        return
    st.caption("Reconstruction from published experiment code — NOT verified against the recordings. Assumes image center (650, 820), native image size, and a 1920 × 1080 screen.")
    with st.container() if nested else st.expander("Image position on the experiment screen"):
        st.caption("Defaults use the published layout. Edit only if you have better experiment-layout evidence; changing values does not verify alignment.")
        columns = st.columns(4)
        values = [column.number_input(label, value=value, min_value=minimum, key=key + label)
                  for column, label, value, minimum in zip(columns,
                    ["Left", "Top", "Displayed width", "Displayed height"],
                    [650.0 - image.width / 2, 820.0 - image.height / 2, float(image.width), float(image.height)],
                    [None, None, 1.0, 1.0])]
        show_points = st.checkbox("Show fixation dots too", key=key + "dots")
    points = image_gaze_points(runs, values, image.size)
    st.plotly_chart(image_gaze_figure(image, points, show_points), use_container_width=True, key=key + "overlay")
    st.caption("Heatmap weights fixation centers by supported duration, smoothed over 20 image pixels. Warm colors indicate greater relative density within this trial, not translation relevance. Out-of-image centers are excluded; this is not calibration or proof of attention to an object.")
    if not points.empty:
        used = points.duration_seconds.sum()
        total = runs.duration_seconds.sum()
        st.caption(f"Shown inside the assumed image: {used:.2f} s of {total:.2f} s supported fixation time. Red is this image's local maximum, even when very little gaze falls inside it.")
    if points.empty:
        st.info("No supported fixation centers fall inside this image rectangle in the selected phase.")


def render_comparison(trials, row, root):
    import streamlit as st
    peers = trials[trials.sentence.eq(row.sentence) & trials.source.eq(row.source)].copy()
    if "included_in_audit" in peers:
        peers = peers[peers.included_in_audit.astype(str).str.lower().eq("true")]
    peers["id"] = peers.apply(trial_id, axis=1)
    peers = peers.drop_duplicates("id")
    st.caption(f"{len(peers)} recorded trials of the same source sentence. Conditions and images may differ; these are not matched counterfactuals.")
    st.dataframe(peers[["participant", "condition", "choice", "initial_translation", "updated_translation"]], hide_index=True)
    if len(peers) < 2:
        st.info("Only one trial of this sentence is in the comparison manifest. Load a broader manifest to compare people.")
        return
    indexed = peers.set_index("id")
    selected_trial = peers[peers.participant.eq(row.participant) & peers.condition.eq(row.condition)
                           & peers.initial_translation.eq(row.initial_translation) & peers.updated_translation.eq(row.updated_translation)]
    defaults = selected_trial.id.head(1).tolist()
    others = peers[~peers.id.isin(defaults)]
    if not others.empty:
        cached = others[others.apply(lambda r: bool(r.get("gaze_See_path", "")) and (root / r.get("gaze_See_path", "")).is_file(), axis=1)]
        defaults += (cached if not cached.empty else others).id.head(1).tolist()
    if not defaults:
        defaults = peers.id.head(2).tolist()
    chosen = st.multiselect("Compare trials (up to 4)", peers.id.tolist(),
                            default=defaults, max_selections=4,
                            format_func=lambda value: f"{indexed.loc[value, 'participant']} · {indexed.loc[value, 'condition']} · order {indexed.loc[value].get('order', '')}",
                            key="compare_" + trial_id(row))
    if not chosen:
        return
    for column, value in zip(st.columns(len(chosen)), chosen):
        peer = indexed.loc[value]
        with column:
            st.subheader(f"{peer.participant} · {peer.condition}")
            st.caption(f"Reported decision: {peer.choice}")
            before, after = diff_html(peer.initial_translation, peer.updated_translation)
            st.markdown("Initial")
            st.markdown(before, unsafe_allow_html=True)
            st.markdown("Final")
            st.markdown(after, unsafe_allow_html=True)
            image = str(peer.get("image_path", ""))
            if not image:
                import re
                match = re.search(r"-i([^/.]+)\.mp3$", str(peer.get("raw_mp3_path", "")))
                image = f"images/{match.group(1)}.jpg" if match else ""
            runs = pd.DataFrame()
            path = str(peer.get("gaze_See_path", ""))
            gaze_file = resolve_ui(root, path)
            if gaze_file is not None:
                _, runs = inspect_gaze(pd.read_csv(gaze_file))
            else:
                st.caption("See gaze is not locally cached; missing data is not absence of attention.")
            image_file = resolve_ui(root, image)
            if image_file is not None:
                render_image_gaze(image_file, runs, "See", "peer_" + value, nested=True)
            else:
                st.caption("Assigned image is not locally cached.")


def render_reviews(row, database, gaze_figure=None):
    import streamlit as st
    import sqlite3
    st.subheader("Evidence notebook")
    st.caption("Manual categories are reviewer judgments, not automatic semantic ground truth. Saving a correction preserves the earlier review in history.")
    try:
        history = read_reviews(database)
    except (OSError, sqlite3.Error) as exc:
        st.error(f"Cannot read the local evidence notebook: {exc}")
        return
    current = latest_reviews(history)
    identifier = trial_id(row)
    reviewer = st.text_input("Reviewer", key="process_reviewer")
    existing = current[(current.trial_id == identifier) & (current.reviewer == reviewer.strip())]
    old = existing.iloc[-1] if not existing.empty else None
    with st.form("review_" + identifier + "_" + reviewer.strip()):
        label = st.selectbox("Change category", LABELS, index=LABELS.index(old.label) if old is not None else LABELS.index("Unclear"))
        confidence = st.selectbox("Review confidence", CONFIDENCES, index=CONFIDENCES.index(old.confidence) if old is not None else 0)
        note = st.text_area("Evidence / uncertainty", value=old.note if old is not None else "",
                            help="Describe the changed meaning or equivalent wording, cite what you checked, and record uncertainties. Do not infer causation from gaze alone.")
        submitted = st.form_submit_button("Save review locally")
    if submitted:
        try:
            save_review(database, row, reviewer, label, confidence, note)
            st.success("Review saved locally; previous versions are retained.")
            history = read_reviews(database)
            current = latest_reviews(history)
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(str(exc))
    this_trial = current[current.trial_id == identifier]
    if not this_trial.empty:
        st.dataframe(this_trial[["reviewer", "label", "confidence", "note", "created_utc"]], hide_index=True)
        if this_trial.label.nunique() > 1:
            st.warning("Reviewers disagree on this trial. Do not treat one category as a consensus label.")
    st.download_button("Download this evidence card (HTML)", evidence_html(row, this_trial, gaze_figure),
                       file_name=f"{row.participant}_{row.sentence}_evidence.html", mime="text/html")
    with st.expander("Review summaries and report exports"):
        a, b = st.columns(2)
        a.metric("Reviewed trials", current.trial_id.nunique())
        b.metric("Latest reviewer judgments", len(current))
        st.caption("Counts below are by reviewer; multiple raters do not create independent experimental trials. Unreviewed trials and previous versions are not counted as current labels.")
        if current.empty:
            st.info("No reviews saved yet.")
        else:
            counts = current.groupby(["reviewer", "label"]).size().reset_index(name="judgments")
            st.dataframe(counts, hide_index=True)
            st.download_button("Download latest reviews (CSV)", safe_csv(export_table(current)), "latest_process_reviews.csv", "text/csv")
            st.download_button("Download review counts (CSV)", safe_csv(counts), "process_review_counts.csv", "text/csv")
            st.download_button("Download complete history (JSON)", export_table(history).to_json(orient="records", force_ascii=False, indent=2), "process_review_history.json", "application/json")
        st.caption(f"Local notebook: {database}")


def clock_seconds(values):
    # EMMT writes exact whole seconds as HH:MM:SS. (empty fractional part).
    normalized = values.astype(str).str.replace(r"\.$", ".0", regex=True)
    return pd.to_timedelta(normalized, errors="coerce").dt.total_seconds()


def inspect_gaze(frame):
    """Summarize fixation runs using elapsed timestamps, never sample counts as time."""
    data = frame.copy()
    data["clock"] = clock_seconds(data.TimeStamp)
    for name in ("X", "Y"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    invalid_time = int(data.clock.isna().sum())
    data = data.dropna(subset=["clock"]).sort_values("clock", kind="stable")
    duplicates = int(data.clock.duplicated().sum())
    data = data.drop_duplicates("clock")
    if len(data) < 2:
        return {"usable": False, "rows": len(data)}, pd.DataFrame()
    dt = data.clock.shift(-1) - data.clock
    # Missing recording spans are not attributed to the previous fixation.
    typical = float(dt[dt > 0].median())
    supported = dt.notna() & (dt > 0) & (dt <= max(.02, 10 * typical))
    xy = np.isfinite(data.X) & np.isfinite(data.Y)
    fix = data.Event.fillna("").str.lower().eq("fixation") & xy
    breaks = (~fix.shift(fill_value=False)) | (~supported.shift(fill_value=False))
    data["run"] = (fix & breaks).cumsum()
    episodes = []
    for _, block in data[fix].groupby("run"):
        ids = block.index
        duration = float(dt.loc[ids].where(supported.loc[ids], 0).sum())
        if duration <= 0:
            continue
        episodes.append({"start_seconds": float(block.clock.iloc[0] - data.clock.iloc[0]),
                         "duration_seconds": duration,
                         "X": float(block.X.median()), "Y": float(block.Y.median())})
    elapsed = float(data.clock.iloc[-1] - data.clock.iloc[0])
    summary = {"usable": True, "rows": len(data), "invalid_timestamps": invalid_time,
               "duplicate_timestamps": duplicates, "recording_span_seconds": elapsed,
               "supported_seconds": float(dt.where(supported, 0).sum()),
               "valid_xy_fraction": float(xy.mean()), "fixation_runs": len(episodes),
               "fixation_seconds": sum(e["duration_seconds"] for e in episodes)}
    return summary, pd.DataFrame(episodes)


def render(shared_selection=None):
    import streamlit as st
    st.title("Human translation process")
    manifest_path = os.getenv("PROCESS_MANIFEST", "")
    root = Path(os.getenv("DATA_PATH", "ufal_emmt"))
    if not manifest_path or not Path(manifest_path).is_file():
        st.info("Set PROCESS_MANIFEST to the prepared trial manifest and DATA_PATH to the local EMMT data folder.")
        return
    trials = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"participant", "sentence", "condition", "choice", "source", "initial_translation", "updated_translation", "gaze_See_path", "image_condition"}
    missing = required - set(trials.columns)
    if missing:
        st.error("Manifest is missing required columns: " + ", ".join(sorted(missing)))
        return
    suggested = Path(manifest_path).parent.parent / "multimodal_audit/trial_manifest.csv"
    comparison_path = os.getenv("PROCESS_COMPARISON_MANIFEST", str(suggested) if suggested.is_file() else "")
    comparison = trials
    if comparison_path:
        try:
            comparison = pd.read_csv(comparison_path, keep_default_na=False)
            if not required - {"image_condition", "gaze_See_path"} <= set(comparison.columns):
                raise ValueError("Comparison manifest does not contain the required transcript columns.")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.warning(f"Comparison manifest could not be used: {exc}. Comparing within the current manifest instead.")
            comparison = trials
    available = trials.copy()
    if available.empty:
        st.warning("No local See gaze files match this manifest.")
        return
    available["case_id"] = available.participant + "_" + available.sentence
    duplicates = available.case_id.duplicated(keep=False)
    if duplicates.any():
        available.loc[duplicates, "case_id"] += available.loc[duplicates].apply(lambda r: "_" + trial_id(r)[:8], axis=1)
    if shared_selection is not None:
        row, phase = shared_selection
        st.caption(f"Shared trial: {row.participant}_{row.sentence} · {row.condition} · {phase} · order {row.get('order', '')}")
    else:
        condition = st.sidebar.selectbox("Image condition", ["All"] + sorted(available.image_condition.unique()))
        if condition != "All":
            available = available[available.image_condition == condition]
        case_id = st.selectbox("Trial", available.case_id.tolist(), key=f"process_trial_{condition}")
        row = available[available.case_id == case_id].iloc[0]
        phase = st.selectbox("Gaze phase", ["See", "Read", "Translate", "Update"])
    st.caption(f"{row.participant} · {row.sentence} · {row.condition} · reported decision: {row.choice}")
    st.subheader("What changed?")
    st.write("English source:", row.source)
    left, right = st.columns(2)
    left.write("Initial Czech translation")
    right.write("Final Czech translation")
    before, after = diff_html(row.initial_translation, row.updated_translation)
    left.markdown(before, unsafe_allow_html=True)
    right.markdown(after, unsafe_allow_html=True)
    st.caption("Red / crossed out: removed or replaced. Green: added or replaced. Highlighting shows text differences only.")
    if "<no translation" in (row.initial_translation + row.updated_translation).lower():
        st.warning("A translation is missing in this trial. Do not classify this as a normal before/after revision.")
    st.info("Text difference is not semantic correction or improved translation quality. Semantic categories require bilingual review.")
    image_path = str(row.get("image_path", ""))
    runs = pd.DataFrame()
    gaze_path = row.get(f"gaze_{phase}_path", "")
    gaze_file = resolve_ui(root, gaze_path)
    if gaze_file is None:
        st.warning(f"{phase} gaze is not cached for this trial.")
    else:
        summary, runs = inspect_gaze(pd.read_csv(gaze_file))
        if summary.get("usable"):
            event_path = row.get("raw_events_path", "")
            event_file = resolve_ui(root, event_path) if phase == "See" else None
            if event_file is not None:
                event_times = {}
                for line in event_file.read_text().splitlines():
                    fields = line.split("\t")
                    if len(fields) > 1:
                        event_times[fields[0]] = clock_seconds(pd.Series([fields[1]])).iloc[0]
                start = event_times.get("present image")
                stop = event_times.get("record translation with image")
                if start is not None and stop is not None and stop > start:
                    span = stop - start
                    if summary["supported_seconds"] < .9 * span:
                        st.warning("Gaze does not cover the full image-viewing phase. Missing time is not evidence that the participant stopped looking.")
        else:
            st.warning("Insufficient gaze samples for this phase.")
    image_file = resolve_ui(root, image_path)
    if image_file is not None:
        render_image_gaze(image_file, runs, phase, "main_" + trial_id(row))
    else:
        st.info("Assigned image is not locally cached.")
    audio_path = row.get("raw_mp3_path", "")
    audio_file = resolve_ui(root, audio_path)
    if audio_file is not None:
        st.audio(str(audio_file), format="audio/mpeg")
        st.caption("Whole-trial audio, not aligned to the phase plot. No pause or self-correction labels have been validated.")
    else:
        st.caption("Whole-trial audio has not been cached for this trial.")
    st.caption(f"Manifest: {len(available)} trials from {available.participant.nunique()} participants. Availability is not quality-control eligibility; browsing does not rerun statistical analyses.")
    with st.expander("Compare people on this sentence"):
        render_comparison(comparison, row, root)
