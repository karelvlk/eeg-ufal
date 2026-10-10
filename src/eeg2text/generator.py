"""
EEG-to-Text Generator
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy import signal
from transformers import GPT2Tokenizer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Sentences.csv loader
# ---------------------------------------------------------------------------

def load_sentences_dataframe(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load ``Sentences.csv`` from ``<data_path>/probes/Sentences.csv``.
    """
    import os
    empty = pd.DataFrame(columns=["sentence_id", "sentence_content"])
    root  = data_path or os.getenv("DATA_PATH", "./ufal_emmt")
    path  = Path(root) / "probes" / "Sentences.csv"
    if not path.exists():
        print(f"WARNING: Sentences.csv not found at {path}")
        return empty
    try:
        df = pd.read_csv(path, names=["sentence_id", "sentence_content"])
        df = df.dropna(subset=["sentence_content"])
        df["sentence_id"]      = df["sentence_id"].astype(str).str.strip()
        df["sentence_content"] = df["sentence_content"].astype(str).str.strip()
        print(f"Loaded {len(df)} sentences from {path}")
        return df.reset_index(drop=True)
    except Exception as exc:
        print(f"ERROR reading {path}: {exc}")
        return empty


# ---------------------------------------------------------------------------
# Text post-processing (mirrors training notebook's clean_generated_text)
# ---------------------------------------------------------------------------

def clean_generated_text(text: str) -> str:
    for sep in ("...", ".."):
        if sep in text:
            text = text.split(sep)[0]

    text = text.rstrip(".")

    for tok in ("<s>", "</s>", "<pad>", "<unk>", "Ġ"):
        text = text.replace(tok, "")

    words   = text.split()
    cleaned = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]
    return " ".join(cleaned).strip()


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------

class RawNetEncoder(nn.Module):
    """
    Multi-scale convolutional encoder followed by a Transformer encoder.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.conv1_large = nn.Sequential(
            nn.Conv1d(config["n_channels"], config["conv1_out"],
                      kernel_size=config["conv1_kernel"],
                      padding=config["conv1_kernel"] // 2),
            nn.BatchNorm1d(config["conv1_out"]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.conv2_small = nn.Sequential(
            nn.Conv1d(config["n_channels"], config["conv2_out"],
                      kernel_size=config["conv2_kernel"],
                      padding=config["conv2_kernel"] // 2),
            nn.BatchNorm1d(config["conv2_out"]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        conv_out_dim = config["conv1_out"] + config["conv2_out"]   # 96
        pooled_time  = config["max_word_samples"] // 2             # 64
        self.projection = nn.Sequential(
            nn.Linear(conv_out_dim * pooled_time, config["d_model"]),
            nn.LayerNorm(config["d_model"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["d_model"] * 4,
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_encoder_layers"]
        )

    def forward(
        self,
        raw_eeg: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        b, s, t, c = raw_eeg.shape
        x    = raw_eeg.reshape(b * s, t, c).permute(0, 2, 1)   # [B*S, C, T]
        feat = torch.cat([self.conv1_large(x), self.conv2_small(x)], dim=1)
        feat = feat.reshape(feat.size(0), -1)
        feat = self.projection(feat).reshape(b, s, -1)
        return self.encoder(feat, src_key_padding_mask=src_key_padding_mask)


class TextDecoder(nn.Module):
    """
    Transformer decoder that attends to the EEG encoder memory.
    """

    def __init__(self, config: dict, vocab_size: int) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, config["d_model"])
        self.pos_embed   = nn.Embedding(config["max_seq_len"], config["d_model"])
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["d_model"] * 4,
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.decoder     = nn.TransformerDecoder(
            decoder_layer, num_layers=config["num_decoder_layers"]
        )
        self.output_proj = nn.Linear(config["d_model"], vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s = tokens.shape
        pos  = torch.arange(s, device=tokens.device).unsqueeze(0).expand(b, -1)
        x    = self.token_embed(tokens) + self.pos_embed(pos)
        out  = self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)


class RawEEGToTextModel(nn.Module):
    """
    Full encoder-decoder model: RawNetEncoder + TextDecoder.
    """

    def __init__(self, config: dict, vocab_size: int) -> None:
        super().__init__()
        self.rawnet_encoder = RawNetEncoder(config)
        self.text_decoder   = TextDecoder(config, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        raw_eeg: torch.Tensor,
        eos_id: int,
        max_len: int = 30,
    ) -> torch.Tensor:
       
        self.eval()
        device     = raw_eeg.device
        batch_size = raw_eeg.size(0)
        pad_id     = eos_id   # GPT-2: pad == eos == 50256

        encoded   = self.rawnet_encoder(raw_eeg)
        generated = torch.full((batch_size, 1), eos_id,
                               dtype=torch.long, device=device)
        finished  = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            n    = generated.size(1)
            mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()

            logits     = self.text_decoder(generated, encoded, tgt_mask=mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            next_token[finished.unsqueeze(1)] = pad_id
            finished  = finished | (next_token.squeeze(1) == eos_id)
            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return generated


# ---------------------------------------------------------------------------
# Model configuration
# (verified against checkpoint — param count = 33,956,593)
# ---------------------------------------------------------------------------

_CONFIG: dict = {
    "n_channels":         4,
    "d_model":            256,
    "nhead":              8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 4,
    "dropout":            0.3,
    "max_seq_len":        50,
    "conv1_kernel":       25,
    "conv1_out":          32,
    "conv2_kernel":       7,
    "conv2_out":          64,
    "max_word_samples":   128,
}

_EEG_CHANNELS: list[str] = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]


# ---------------------------------------------------------------------------
# Public generator class
# ---------------------------------------------------------------------------

class EEG2TextGenerator:
    """
    High-level interface for EEG-to-text generation.
    """

    def __init__(
        self,
        model_id: str = "sajjad5221/eeg2text",
        device: str = "auto",
        data_path: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
        )

        print("Loading GPT-2 tokenizer…")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Building model…")
        self.model = RawEEGToTextModel(_CONFIG, len(self.tokenizer)).to(self.device)
        self._load_weights(model_id)

        self._data_path = data_path
        self._sentences_df: Optional[pd.DataFrame] = None

        # HuggingFace dataset (optional — speeds up lookups dramatically)
        self.hf_dataset = None
        try:
            from datasets import load_dataset
            print("Fetching HF dataset…")
            self.hf_dataset = load_dataset("sajjad5221/eeg2text-emmt-dataset")
        except Exception as exc:
            print(f"HF dataset unavailable ({exc}).  CSV fallback active.")

        print(f"EEG2TextGenerator ready  [{self.device}]")

    # ── Weight loading ────────────────────────────────────────────────────────

    def _load_weights(self, model_id: str) -> None:
        """Download and load model weights from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=model_id, filename="best_rawnet_model.pt")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = (
                ckpt["model_state_dict"]
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                else ckpt
            )
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if not missing and not unexpected:
                print("SUCCESS: Weights loaded — perfect match.")
            else:
                if unexpected:
                    print(f"WARNING: Unexpected keys: {unexpected[:3]}")
                if missing:
                    print(f"WARNING: Missing keys:    {missing[:3]}")
                print("Weights loaded (partial match).")
        except Exception as exc:
            print(f"ERROR loading weights: {exc}")

    # ── Dataset helpers ───────────────────────────────────────────────────────

    def get_dataset_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame of all available sentences.
        """
        if self._sentences_df is None:
            self._sentences_df = load_sentences_dataframe(self._data_path)

        if self.hf_dataset is not None:
            try:
                rows = []
                for split in self.hf_dataset.keys():
                    for sample in self.hf_dataset[split]:
                        rows.append({
                            "sentence_id":      sample.get("sentence_id", ""),
                            "sentence_content": sample.get("sentence_text", ""),
                            "participant_id":   sample.get("participant_id", "Unknown"),
                        })
                hf_df = (
                    pd.DataFrame(rows)
                    .drop_duplicates(subset=["sentence_content"])
                    .reset_index(drop=True)
                )
                if not hf_df.empty:
                    return hf_df
            except Exception:
                pass

        return self._sentences_df.copy()

    # ── EEG preprocessing (mirrors training notebook) ─────────────────────────

    def _preprocess_full_signal(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Preprocess a full sentence-length EEG signal.
        """
        N_CH = _CONFIG["n_channels"]
        SR   = 256

        try:
            eeg = np.array(eeg_data, dtype=np.float64)
            eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

            preprocessed = np.zeros_like(eeg)

            if eeg.shape[0] >= 30:
                b_notch, a_notch = signal.iirnotch(50, 30, SR)
                nyq = SR / 2
                b_band, a_band  = signal.butter(4, [0.5 / nyq, 50.0 / nyq], btype="band")
                for ch in range(min(eeg.shape[1], N_CH)):
                    col = eeg[:, ch]
                    try:
                        col = signal.filtfilt(b_notch, a_notch, col)
                        col = signal.filtfilt(b_band,  a_band,  col)
                    except Exception:
                        pass
                    preprocessed[:, ch] = col
            else:
                preprocessed = eeg.copy()

            # Per-channel robust normalisation: (x − median) / IQR
            for ch in range(min(preprocessed.shape[1], N_CH)):
                col    = preprocessed[:, ch]
                median = np.median(col)
                q75, q25 = np.percentile(col, [75, 25])
                iqr = q75 - q25
                preprocessed[:, ch] = (col - median) / iqr if iqr > 1e-6 else col - median

            return np.clip(preprocessed, -5.0, 5.0).astype(np.float32)

        except Exception:
            return np.zeros_like(eeg_data, dtype=np.float32)

    @staticmethod
    def _pad_or_truncate(segment: np.ndarray, max_t: int, n_ch: int) -> np.ndarray:
        """Pad or truncate ``(time, channels)`` to ``(max_t, n_ch)``."""
        seg = segment[:, :n_ch]
        if len(seg) >= max_t:
            return seg[:max_t].astype(np.float32)
        pad = np.zeros((max_t - len(seg), n_ch), dtype=np.float32)
        return np.vstack([seg, pad]).astype(np.float32)

    # ── Tensor construction ───────────────────────────────────────────────────

    def _build_tensor_from_sample(
        self,
        sample: dict,
        sentence_text: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build the ``raw_eeg`` tensor from an already-loaded HF dataset sample.
        """
        MAX_T = _CONFIG["max_word_samples"]
        N_CH  = _CONFIG["n_channels"]

        try:
            txt        = sentence_text or sample.get("sentence_text", "")
            num_tokens = max(1, len(self.tokenizer.encode(txt, add_special_tokens=True)))

            segs = sample.get("word_eeg_segments", [])
            if not segs:
                return None

            windows = [self._pad_or_truncate(np.array(s, dtype=np.float32), MAX_T, N_CH)
                       for s in segs]
            raw_eeg = np.stack(windows, axis=0)   # (num_words, 128, 4)

            # Align to token count
            if len(raw_eeg) > num_tokens:
                raw_eeg = raw_eeg[:num_tokens]
            elif len(raw_eeg) < num_tokens:
                pad     = np.zeros((num_tokens - len(raw_eeg), MAX_T, N_CH), dtype=np.float32)
                raw_eeg = np.concatenate([raw_eeg, pad], axis=0)

            return torch.from_numpy(raw_eeg).float().unsqueeze(0).to(self.device)

        except Exception as exc:
            print(f"_build_tensor_from_sample failed: {exc}")
            return None

    def _build_tensor_from_hf(
        self,
        sentence_id: str,
        participant_id: str,
    ) -> Optional[torch.Tensor]:
        
        if self.hf_dataset is None:
            return None

        sid = sentence_id.upper()
        pid = participant_id.upper()

        try:
            for split in self.hf_dataset.keys():
                for sample in self.hf_dataset[split]:
                    if (str(sample.get("sentence_id",    "")).upper() == sid and
                            str(sample.get("participant_id", "")).upper() == pid):
                        txt = sample.get("sentence_text", "")
                        return self._build_tensor_from_sample(sample, txt)
        except Exception as exc:
            print(f"HF dataset scan failed: {exc}")

        return None

    def _build_tensor_from_csv(
        self,
        eeg_dict: dict,
        sentence_text: Optional[str],
    ) -> torch.Tensor:
       
        MAX_T = _CONFIG["max_word_samples"]
        N_CH  = _CONFIG["n_channels"]

        eeg_raw   = np.stack([eeg_dict[ch] for ch in _EEG_CHANNELS], axis=1).astype(np.float32)
        n_samples = eeg_raw.shape[0]
        eeg_proc  = self._preprocess_full_signal(eeg_raw)

        if sentence_text:
            num_words  = max(1, len(sentence_text.split()))
            num_tokens = max(1, len(self.tokenizer.encode(sentence_text, add_special_tokens=True)))
        else:
            num_words  = 8
            num_tokens = 10

        num_words  = min(num_words,  _CONFIG["max_seq_len"])
        num_tokens = min(num_tokens, _CONFIG["max_seq_len"])

        if n_samples < num_words:
            eeg_proc  = np.tile(eeg_proc, (-(-num_words // n_samples), 1))
            n_samples = eeg_proc.shape[0]

        seg_len = n_samples // num_words
        windows = [
            self._pad_or_truncate(eeg_proc[i * seg_len: (i + 1) * seg_len], MAX_T, N_CH)
            for i in range(num_words)
        ]
        raw_eeg = np.stack(windows, axis=0)

        if len(raw_eeg) > num_tokens:
            raw_eeg = raw_eeg[:num_tokens]
        elif len(raw_eeg) < num_tokens:
            pad     = np.zeros((num_tokens - len(raw_eeg), MAX_T, N_CH), dtype=np.float32)
            raw_eeg = np.concatenate([raw_eeg, pad], axis=0)

        return torch.from_numpy(raw_eeg).float().unsqueeze(0).to(self.device)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_from_sentence(
        self,
        eeg_dict: Optional[dict] = None,
        sentence_text: Optional[str] = None,
        sentence_id:   Optional[str] = None,
        participant_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, float]:
        """
        Generate text from EEG data.
        """
        try:
            eeg_tensor = None

            # Extract metadata injected by the loader
            if eeg_dict:
                sentence_id    = eeg_dict.pop("_sentence_id",    sentence_id)
                participant_id = eeg_dict.pop("_participant_id", participant_id)
                hf_sample      = eeg_dict.pop("_hf_sample",      None)
                eeg_dict.pop("_hf_split", None)
                eeg_dict.pop("_hf_idx",   None)
            else:
                hf_sample = None

            # Path 1: HF sample passed directly
            if hf_sample is not None:
                eeg_tensor = self._build_tensor_from_sample(hf_sample, sentence_text)
                if eeg_tensor is not None:
                    print(f"Using embedded HF sample  {participant_id}/{sentence_id}")

            # Path 2: HF dataset scan
            if eeg_tensor is None and sentence_id and participant_id:
                eeg_tensor = self._build_tensor_from_hf(sentence_id, participant_id)
                if eeg_tensor is not None:
                    print(f"Using HF scan result  {participant_id}/{sentence_id}")

            # Path 3: CSV fallback
            if eeg_tensor is None:
                if not eeg_dict:
                    return "Error: No EEG data provided.", 0.0
                eeg_tensor = self._build_tensor_from_csv(eeg_dict, sentence_text)

            eos_id    = self.tokenizer.eos_token_id
            token_ids = self.model.generate(eeg_tensor, eos_id)
            raw_text  = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
            return clean_generated_text(raw_text), 1.0

        except Exception as exc:
            return f"Generation error: {exc}", 0.0