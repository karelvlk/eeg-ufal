"""
EEG-to-Text Generation Page
============================
Streamlit page that loads the trained RawNet model and generates text from
preprocessed EEG signals retrieved directly from the HuggingFace dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.eeg2text.generator import EEG2TextGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_overlap(text1: str, text2: str) -> float:
    """
    Jaccard word-overlap score between two strings.

    Returns a float in [0, 1] where 1 means identical word sets.
    Used as a quick proxy metric; not a substitute for BLEU.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_eeg2text_page(
    sentences_df: pd.DataFrame,
    eeg_data_loader,
    gaze_data_loader,
) -> None:
    """
    Render the EEG-to-Text generation UI.
    """
    st.title("🧠 EEG-to-Text Generation")
    st.markdown(
        "Generate text directly from EEG brain signals using a trained "
        "[RawNet](https://huggingface.co/sajjad5221/eeg2text) model. "
        "EEG features are loaded from the "
        "[EMMT HuggingFace dataset](https://huggingface.co/datasets/sajjad5221/eeg2text-emmt-dataset)."
    )

    # ── 1. Load model (cached for session lifetime) ───────────────────────────
    @st.cache_resource(show_spinner="Loading EEG2Text model…")
    def _load_model():
        return EEG2TextGenerator(model_id="sajjad5221/eeg2text", device="auto")

    try:
        generator = _load_model()
        st.success("✓ Model loaded successfully")
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    # ── 2. Sentence selector ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Select Data")

    if sentences_df is None or sentences_df.empty:
        st.error("No sentences available.  Check the HuggingFace dataset connection.")
        return

    sentences_df = sentences_df.copy()
    sentences_df["_display"] = sentences_df.apply(
        lambda r: f"[{r['sentence_id']}] {str(r['sentence_content'])[:45]}…", axis=1
    )

    selected_label = st.selectbox("Select Sentence", sentences_df["_display"].unique())
    selected_row   = sentences_df[sentences_df["_display"] == selected_label].iloc[0]
    selected_sid   = selected_row["sentence_id"]
    selected_text  = selected_row["sentence_content"]

    st.info(f"**Reference Text:** {selected_text}")

    # ── 3. Generate ───────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🚀 Generate Text from EEG", type="primary", use_container_width=True):
        with st.spinner("Loading EEG features…"):
            eeg_data = eeg_data_loader(selected_sid, None)

        if eeg_data is None:
            st.warning("EEG data unavailable — using zero-signal fallback.")
            eeg_data = {ch: np.zeros(512, dtype=np.float32)
                        for ch in ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]}

        with st.spinner("Generating text from EEG…"):
            generated_text, _ = generator.generate_from_sentence(
                eeg_dict=eeg_data,
                sentence_text=selected_text,
            )

        # ── 4. Results ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Reference (Ground Truth):**")
            st.info(selected_text)
        with col2:
            st.markdown("**Model Output:**")
            st.success(generated_text)

        score = _word_overlap(selected_text, generated_text)
        st.metric("Word Overlap Accuracy", f"{score:.2%}")
