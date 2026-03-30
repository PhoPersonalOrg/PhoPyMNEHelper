"""Timeline-oriented continuous EEG spectrogram from ``mne.io.Raw``.

See ``phopymnehelper/analysis/COMPUTATIONS_README.md`` for the computations contract.
This module centralizes default STFT parameters used by pyPhoTimeline and delegates
FFT work to :class:`phopymnehelper.EEG_data.EEGComputations`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import mne

from phopymnehelper.EEG_data import EEGComputations

DEFAULT_SPECTROGRAM_NPERSEG = 1024
DEFAULT_SPECTROGRAM_NOVERLAP = 512

__all__ = ["DEFAULT_SPECTROGRAM_NPERSEG", "DEFAULT_SPECTROGRAM_NOVERLAP", "compute_raw_eeg_spectrogram"]


def compute_raw_eeg_spectrogram(raw: mne.io.Raw, *, nperseg: int = DEFAULT_SPECTROGRAM_NPERSEG, noverlap: int = DEFAULT_SPECTROGRAM_NOVERLAP, picks: Any = None, mask_bad_annotated_times: bool = True) -> Dict[str, Any]:
    """Compute continuous per-channel spectrogram; same defaults as timeline XDF processing."""
    return EEGComputations.raw_spectogram_working(raw, picks=picks, nperseg=nperseg, noverlap=noverlap, mask_bad_annotated_times=mask_bad_annotated_times)
