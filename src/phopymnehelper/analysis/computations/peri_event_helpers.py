"""Peri-event spectrogram slicing, band power, and baseline dB normalization.

Pure-numpy helpers that operate on pre-computed spectrogram arrays.
Originally from ``pyPhoTimeline.analysis.peri_event``; inlined here so
``PhoPyMNEHelper`` nodes can use them without a ``pyPhoTimeline`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_PRE_S = 600.0
DEFAULT_POST_S = 1800.0

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class PeriEventSlice:
    """One event's spectrogram on a relative-time grid."""

    t_rel: np.ndarray  # (n_times,)
    freqs: np.ndarray  # (n_freqs,)
    Sxx: np.ndarray  # (n_channels, n_freqs, n_times) with NaN outside coverage
    ch_names: List[str]
    coverage_pre_s: float
    coverage_post_s: float
    event_t_unix: float



class PeriEventHelpers:

    @classmethod
    def relative_time_grid(cls, pre_s: float = DEFAULT_PRE_S, post_s: float = DEFAULT_POST_S, hop_s: float = None) -> np.ndarray:
        """Build inclusive-ish relative-time centers from -pre_s to +post_s at hop_s."""
        if hop_s is None:
            raise ValueError("hop_s argument is required")
        hop = float(hop_s)
        if hop <= 0:
            raise ValueError("hop_s must be positive")
        n_pre = int(np.floor(pre_s / hop))
        n_post = int(np.floor(post_s / hop))
        return np.arange(-n_pre, n_post + 1, dtype=np.float64) * hop


    @classmethod
    def infer_hop_seconds(cls, t_unix: np.ndarray) -> float:
        t = np.asarray(t_unix, dtype=np.float64)
        if t.size < 2:
            return 1.0
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 1.0
        return float(np.median(diffs))


    @classmethod
    def slice_peri_event_spectrogram(cls, Sxx: np.ndarray, freqs: np.ndarray, t_unix: np.ndarray, ch_names: Sequence[str], event_t_unix: float, pre_s: float = DEFAULT_PRE_S, post_s: float = DEFAULT_POST_S, hop_s: Optional[float] = None) -> PeriEventSlice:
        """Map session spectrogram onto a relative grid; missing bins stay NaN."""
        Sxx_arr = np.asarray(Sxx, dtype=np.float32)
        freqs_arr = np.asarray(freqs, dtype=np.float64)
        t_arr = np.asarray(t_unix, dtype=np.float64)
        hop = float(hop_s) if hop_s is not None else cls.infer_hop_seconds(t_arr)
        t_rel = cls.relative_time_grid(pre_s=pre_s, post_s=post_s, hop_s=hop)
        n_ch, n_f, _n_t = Sxx_arr.shape
        out = np.full((n_ch, n_f, t_rel.size), np.nan, dtype=np.float32)

        if t_arr.size == 0:
            return PeriEventSlice(
                t_rel=t_rel,
                freqs=freqs_arr,
                Sxx=out,
                ch_names=list(ch_names),
                coverage_pre_s=0.0,
                coverage_post_s=0.0,
                event_t_unix=float(event_t_unix),
            )

        # Nearest-neighbor assign within half-hop
        half = 0.5 * hop
        t_target = float(event_t_unix) + t_rel
        for i, tt in enumerate(t_target):
            j = int(np.argmin(np.abs(t_arr - tt)))
            if abs(float(t_arr[j]) - float(tt)) <= half + 1e-9:
                out[:, :, i] = Sxx_arr[:, :, j]

        finite_t = t_rel[np.any(np.isfinite(out), axis=(0, 1))]
        if finite_t.size == 0:
            cov_pre = 0.0
            cov_post = 0.0
        else:
            pre_bins = finite_t[finite_t <= 0]
            post_bins = finite_t[finite_t >= 0]
            cov_pre = float(max(0.0, -float(pre_bins.min()))) if pre_bins.size else 0.0
            cov_post = float(max(0.0, float(post_bins.max()))) if post_bins.size else 0.0

        return PeriEventSlice(
            t_rel=t_rel,
            freqs=freqs_arr,
            Sxx=out,
            ch_names=list(ch_names),
            coverage_pre_s=cov_pre,
            coverage_post_s=cov_post,
            event_t_unix=float(event_t_unix),
        )


    @classmethod
    def mean_sxx_over_channels(cls, Sxx: np.ndarray, channel_indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return (n_freqs, n_times) nanmean over channels."""
        arr = np.asarray(Sxx, dtype=np.float64)
        if channel_indices is not None:
            arr = arr[list(channel_indices), ...]
        with np.errstate(all="ignore"):
            return np.nanmean(arr, axis=0)


    @classmethod
    def band_power_curves(cls, Sxx_2d: np.ndarray, freqs: np.ndarray, bands: Optional[Mapping[str, Tuple[float, float]]] = None) -> Dict[str, np.ndarray]:
        """Collapse (n_freqs, n_times) into band mean power vs time."""
        bands = bands or DEFAULT_BANDS
        freqs_arr = np.asarray(freqs, dtype=np.float64)
        S = np.asarray(Sxx_2d, dtype=np.float64)
        out: Dict[str, np.ndarray] = {}
        for name, (f_lo, f_hi) in bands.items():
            mask = (freqs_arr >= float(f_lo)) & (freqs_arr < float(f_hi))
            if not np.any(mask):
                out[name] = np.full(S.shape[-1], np.nan, dtype=np.float64)
            else:
                with np.errstate(all="ignore"):
                    out[name] = np.nanmean(S[mask, :], axis=0)
        ## END for name, (f_lo, f_hi) in bands.items()...
        return out


    @classmethod
    def baseline_normalize_db(cls, power: np.ndarray, t_rel: np.ndarray, baseline_pre_s: float = DEFAULT_PRE_S, baseline_post_s: float = 0.0) -> np.ndarray:
        """Return 10*log10(power / baseline_mean) with NaN where baseline unavailable.

        Works for 1D (time,) or 2D (freq, time) arrays.
        """
        arr = np.asarray(power, dtype=np.float64)
        t = np.asarray(t_rel, dtype=np.float64)
        base_mask = (t >= -float(baseline_pre_s)) & (t <= float(baseline_post_s))
        if arr.ndim == 1:
            baseline = np.nanmean(arr[base_mask]) if np.any(base_mask) else np.nan
            if not np.isfinite(baseline) or baseline <= 0:
                return np.full_like(arr, np.nan, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                return 10.0 * np.log10(arr / baseline)
        if arr.ndim == 2:
            baseline = (
                np.nanmean(arr[:, base_mask], axis=1, keepdims=True)
                if np.any(base_mask)
                else np.full((arr.shape[0], 1), np.nan)
            )
            bad = ~np.isfinite(baseline) | (baseline <= 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = 10.0 * np.log10(arr / baseline)
            out[np.broadcast_to(bad, out.shape)] = np.nan
            return out
        raise ValueError(f"power must be 1D or 2D; got shape {arr.shape}")


    @classmethod
    def channel_indices_for_names(cls, ch_names: Sequence[str], wanted: Optional[Sequence[str]]) -> Optional[List[int]]:
        if wanted is None:
            return None
        name_to_i = {str(n): i for i, n in enumerate(ch_names)}
        idxs = [name_to_i[n] for n in wanted if n in name_to_i]
        return idxs if idxs else None


__all__ = [
    "DEFAULT_BANDS",
    "DEFAULT_POST_S",
    "DEFAULT_PRE_S",
    "PeriEventHelpers",
    "PeriEventSlice",
]
