"""Pure NumPy/SciPy helpers for EEG band-limited Global Field Power (GFP).

Used by :mod:`stream_viewer.renderers.line_power_vis` and timeline detail renderers.
No Qt or pyqtgraph dependencies.

Computation layout
------------------
1. Band-pass each channel (Butterworth SOS, zero-phase ``sosfiltfilt``).
2. GFP trace = sum of squares across channels at each sample.
3. Optional baseline correction: divide by mean GFP over ``[baseline_start, baseline_end]`` and subtract 1.
4. Optional bootstrap CI over channel resampling (expensive for large ``n_bootstrap``).

"""

from __future__ import annotations

from typing import Any, Callable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

Tuple3 = Tuple[int, int, int]

STANDARD_EEG_FREQUENCY_BANDS: List[Tuple[str, float, float]] = [
    ("Theta", 4, 7),
    ("Alpha", 8, 12),
    ("Beta", 13, 25),
    ("Gamma", 30, 45),
]

BAND_COLORS_RGB: List[Tuple3] = [
    (0, 255, 127),
    (0, 191, 191),
    (0, 127, 255),
    (0, 63, 255),
]

CacheKey = Tuple[float, float, float]


def get_filter_sos(fmin: float, fmax: float, srate: float, filter_order: int, cache: MutableMapping[CacheKey, np.ndarray]) -> Optional[np.ndarray]:
    """Return cached second-order-sections for a band-pass, or compute and store."""
    key: CacheKey = (float(fmin), float(fmax), float(srate))
    if key not in cache:
        nyq = srate / 2.0
        low = fmin / nyq
        high = min(fmax / nyq, 0.99)
        if low >= high or low <= 0:
            return None
        try:
            cache[key] = signal.butter(filter_order, [low, high], btype="band", output="sos")
        except ValueError:
            return None
    return cache.get(key)


def bandpass_filter_channels(data: np.ndarray, fmin: float, fmax: float, srate: float, filter_order: int, cache: MutableMapping[CacheKey, np.ndarray]) -> np.ndarray:
    """Band-pass filter multi-channel data (n_channels, n_samples)."""
    sos = get_filter_sos(fmin, fmax, srate, filter_order, cache)
    if sos is None:
        return data
    filtered = np.zeros_like(data)
    for ch_idx in range(data.shape[0]):
        ch_data = data[ch_idx, :]
        valid_mask = np.isfinite(ch_data)
        if not np.any(valid_mask):
            filtered[ch_idx, :] = 0
            continue
        if np.all(valid_mask):
            try:
                filtered[ch_idx, :] = signal.sosfiltfilt(sos, ch_data)
            except ValueError:
                filtered[ch_idx, :] = ch_data
        else:
            interp_data = ch_data.copy()
            valid_indices = np.where(valid_mask)[0]
            invalid_indices = np.where(~valid_mask)[0]
            if len(valid_indices) >= 2:
                interp_data[invalid_indices] = np.interp(invalid_indices, valid_indices, ch_data[valid_indices])
                try:
                    filtered[ch_idx, :] = signal.sosfiltfilt(sos, interp_data)
                except ValueError:
                    filtered[ch_idx, :] = interp_data
            else:
                filtered[ch_idx, :] = 0
    return filtered


def global_field_power(data: np.ndarray) -> np.ndarray:
    """GFP = sum of squares across channels; ``data`` shape (n_channels, n_samples)."""
    return np.sum(data ** 2, axis=0)


def baseline_rescale(gfp: np.ndarray, t_vec: np.ndarray, baseline_start: Optional[float], baseline_end: Optional[float]) -> np.ndarray:
    """Match :class:`stream_viewer.renderers.line_power_vis.LinePowerVis` baseline correction."""
    if baseline_start is None and baseline_end is None:
        return gfp
    start_t = baseline_start if baseline_start is not None else float(t_vec[0])
    end_t = baseline_end if baseline_end is not None else float(t_vec[-1])
    baseline_mask = (t_vec >= start_t) & (t_vec <= end_t)
    if not np.any(baseline_mask):
        return gfp
    baseline_mean = np.nanmean(gfp[baseline_mask])
    if baseline_mean != 0 and np.isfinite(baseline_mean):
        return gfp / baseline_mean - 1
    return gfp


def bootstrap_gfp_ci(data: np.ndarray, n_bootstrap: int, stat_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Channel-bootstrap 95% CI for the GFP-like statistic (default: sum of squares per time)."""
    if stat_fun is None:
        def _default_stat(x: np.ndarray) -> np.ndarray:
            return np.sum(x ** 2, axis=0)

        stat_fun = _default_stat
    n_channels = data.shape[0]
    n_samples = data.shape[1]
    n_bootstrap = max(1, int(n_bootstrap))
    if rng is None:
        rng = np.random.default_rng()
    boot_stats = np.zeros((n_bootstrap, n_samples))
    for i in range(n_bootstrap):
        indices = rng.integers(0, n_channels, size=n_channels)
        boot_data = data[indices, :]
        boot_stats[i, :] = stat_fun(boot_data)
    ci_lower = np.percentile(boot_stats, 2.5, axis=0)
    ci_upper = np.percentile(boot_stats, 97.5, axis=0)
    return ci_lower, ci_upper


def estimate_sample_rate_from_t(t_vec: np.ndarray, fallback: float = 128.0) -> float:
    """Median sample spacing from a 1-D time column (seconds)."""
    if t_vec.size < 2:
        return float(fallback)
    dt = float(np.median(np.diff(t_vec.astype(float))))
    if not np.isfinite(dt) or dt <= 0:
        return float(fallback)
    return 1.0 / dt


def dataframe_to_channel_matrix(detail_df: Any, channel_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (n_channels, n_samples) array and time vector from a sorted detail DataFrame.

    Expects columns ``t`` and all ``channel_names`` present.
    """
    import pandas as pd

    if not isinstance(detail_df, pd.DataFrame) or "t" not in detail_df.columns:
        return np.zeros((0, 0), dtype=float), np.array([], dtype=float)
    df = detail_df.sort_values("t")
    t_vec = df["t"].to_numpy(dtype=float, copy=False)
    cols = [c for c in channel_names if c in df.columns]
    if not cols:
        return np.zeros((0, len(t_vec)), dtype=float), t_vec
    mat = df[list(cols)].to_numpy(dtype=float, copy=False).T
    return mat, t_vec
