"""
EEG2Text sub-package
====================
Text generation from raw EEG signals using a pretrained RawNet transformer.

Public API
----------
EEG2TextGenerator
    High-level generator class.  Load once (expensive), call
    ``generate_from_sentence`` repeatedly.

RawNet  (alias: RawEEGToTextModel)
    The full encoder-decoder architecture.  Exposed for users who want to
    load the checkpoint themselves.

Example
-------
>>> from src.eeg2text import EEG2TextGenerator
>>> gen = EEG2TextGenerator()
>>> text, conf = gen.generate_from_sentence(eeg_dict=my_eeg, sentence_text="…")
"""

from .generator import EEG2TextGenerator, RawEEGToTextModel as RawNet

__all__ = ["EEG2TextGenerator", "RawNet"]
