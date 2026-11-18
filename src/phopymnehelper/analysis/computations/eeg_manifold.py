"""Quality- and MNE-gated EEG activity manifold (2–5D) via PCA / IncrementalPCA / UMAP.

Pure NumPy/SciPy/sklearn helpers with no Qt dependencies. Builds sliding-window feature
vectors from quality-masked EEG (band-power by default, or raw window stats), drops
session-level bad channels from ``time_independent_bad_channels``, and projects to a
low-dimensional embedding that can evolve over time (rolling / incremental PCA).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from phopymnehelper.analysis.computations.gfp_band_power import GFPBandPower, STANDARD_EEG_FREQUENCY_BANDS

logger = logging.getLogger(__name__)

FeatureMode = Literal["band_power", "raw_stats"]
ReducerMethod = Literal["pca", "incremental_pca", "umap"]

DEFAULT_N_COMPONENTS: int = 3
DEFAULT_FEATURE_MODE: FeatureMode = "band_power"
DEFAULT_MIN_QUALITY_TO_KEEP: float = 2.0
DEFAULT_WINDOW_SEC: float = 10.0
DEFAULT_HOP_SEC: float = 0.5
DEFAULT_FIT_HISTORY_SEC: float = 60.0
DEFAULT_FILTER_ORDER: int = 4
DEFAULT_MAX_NAN_FRACTION: float = 0.35
DEFAULT_MIN_CHANNELS: int = 4
DEFAULT_UMAP_N_NEIGHBORS: int = 15
DEFAULT_UMAP_MIN_DIST: float = 0.1

MANIFOLD_COMPONENT_PREFIX: str = "PC"


@dataclass
class EEGManifoldConfig:
    """Configurable parameters for the EEG activity manifold."""

    n_components: int = DEFAULT_N_COMPONENTS
    feature_mode: FeatureMode = DEFAULT_FEATURE_MODE
    method: ReducerMethod = "pca"
    min_quality_to_keep: float = DEFAULT_MIN_QUALITY_TO_KEEP
    mask_unknown_quality: bool = True
    window_sec: float = DEFAULT_WINDOW_SEC
    hop_sec: float = DEFAULT_HOP_SEC
    fit_history_sec: float = DEFAULT_FIT_HISTORY_SEC
    filter_order: int = DEFAULT_FILTER_ORDER
    bands: List[Tuple[str, float, float]] = field(default_factory=lambda: list(STANDARD_EEG_FREQUENCY_BANDS))
    max_nan_fraction: float = DEFAULT_MAX_NAN_FRACTION
    min_channels: int = DEFAULT_MIN_CHANNELS
    session_bad_channels: List[str] = field(default_factory=list)
    allow_umap: bool = True
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST
    live_mode: bool = False
    random_state: int = 0


    def validated(self) -> "EEGManifoldConfig":
        """Return a copy with clamped / sanitized fields."""
        n_comp = int(max(2, min(5, int(self.n_components))))
        method: ReducerMethod = self.method
        if self.live_mode and method == "umap":
            method = "incremental_pca"
        if method not in ("pca", "incremental_pca", "umap"):
            method = "pca"
        if method == "umap" and not self.allow_umap:
            method = "pca"
        feature_mode: FeatureMode = self.feature_mode if self.feature_mode in ("band_power", "raw_stats") else "band_power"
        bands = list(self.bands) if self.bands else list(STANDARD_EEG_FREQUENCY_BANDS)
        return replace(
            self,
            n_components=n_comp,
            method=method,
            feature_mode=feature_mode,
            min_quality_to_keep=float(self.min_quality_to_keep),
            window_sec=max(0.25, float(self.window_sec)),
            hop_sec=max(0.05, float(self.hop_sec)),
            fit_history_sec=max(float(self.window_sec), float(self.fit_history_sec)),
            filter_order=max(1, int(self.filter_order)),
            bands=bands,
            max_nan_fraction=float(np.clip(self.max_nan_fraction, 0.0, 1.0)),
            min_channels=max(1, int(self.min_channels)),
            session_bad_channels=[str(c) for c in (self.session_bad_channels or [])],
            umap_n_neighbors=max(2, int(self.umap_n_neighbors)),
            umap_min_dist=float(max(0.0, self.umap_min_dist)),
        )


    def component_names(self) -> List[str]:
        cfg = self.validated()
        return [f"{MANIFOLD_COMPONENT_PREFIX}{i + 1}" for i in range(cfg.n_components)]


def manifold_track_key_for_eeg_name(eeg_name: str) -> str:
    return f"{eeg_name}_Manifold"


def component_column_names(n_components: int) -> List[str]:
    n = int(max(2, min(5, int(n_components))))
    return [f"{MANIFOLD_COMPONENT_PREFIX}{i + 1}" for i in range(n)]


def resolve_usable_channels(channel_names: Sequence[str], session_bad_channels: Sequence[str]) -> List[str]:
    """Drop session-level bad channels while preserving input order."""
    bad = {str(c) for c in (session_bad_channels or [])}
    return [c for c in channel_names if c not in bad]


def apply_quality_and_session_gates(eeg_df: pd.DataFrame, quality_df: Optional[pd.DataFrame], channel_names: Sequence[str], config: EEGManifoldConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Return quality-masked EEG restricted to usable channels (session bads removed)."""
    cfg = config.validated()
    usable = resolve_usable_channels(channel_names, cfg.session_bad_channels)
    usable = [c for c in usable if c in eeg_df.columns]
    if not usable or eeg_df is None or len(eeg_df) == 0 or "t" not in eeg_df.columns:
        return eeg_df.iloc[0:0].copy() if eeg_df is not None else pd.DataFrame(columns=["t"]), []
    keep_cols = ["t"] + usable
    gated = eeg_df[keep_cols].copy()
    if quality_df is not None and len(quality_df) > 0 and "t" in quality_df.columns:
        from scipy.interpolate import interp1d

        q_cols = [c for c in usable if c in quality_df.columns]
        if q_cols:
            q_sorted = quality_df[["t"] + q_cols].sort_values("t").drop_duplicates(subset=["t"], keep="last")
            x_all = q_sorted["t"].to_numpy(dtype=float)
            target_t = gated["t"].to_numpy(dtype=float)
            for ch in q_cols:
                col_y = q_sorted[ch].to_numpy(dtype=float)
                finite = np.isfinite(col_y)
                if not np.any(finite):
                    gated[ch] = np.nan
                    continue
                xf, yf = x_all[finite], col_y[finite]
                if len(xf) == 1:
                    q_interp = np.full(len(target_t), float(yf[0]), dtype=float)
                else:
                    f_interp = interp1d(xf, yf, kind="linear", fill_value="extrapolate", assume_sorted=True)
                    q_interp = np.asarray(f_interp(target_t), dtype=float)
                is_valid = q_interp >= float(cfg.min_quality_to_keep)
                if cfg.mask_unknown_quality:
                    is_valid = is_valid & np.isfinite(q_interp)
                gated.loc[~is_valid, ch] = np.nan
            ## END for ch in q_cols...
    return gated, usable


def _window_centers(t_vec: np.ndarray, window_sec: float, hop_sec: float, t_min: Optional[float] = None, t_max: Optional[float] = None) -> np.ndarray:
    if t_vec.size < 2:
        return np.array([], dtype=float)
    lo = float(np.nanmin(t_vec)) if t_min is None else float(t_min)
    hi = float(np.nanmax(t_vec)) if t_max is None else float(t_max)
    half = 0.5 * float(window_sec)
    start = lo + half
    end = hi - half
    if end < start:
        return np.array([], dtype=float)
    return np.arange(start, end + 0.5 * hop_sec, float(hop_sec), dtype=float)


def _channel_window_ok(ch_data: np.ndarray, max_nan_fraction: float) -> bool:
    if ch_data.size == 0:
        return False
    nan_frac = float(np.mean(~np.isfinite(ch_data)))
    return nan_frac <= float(max_nan_fraction)


def _band_power_features_for_window(data_ch_x_t: np.ndarray, sfreq: float, bands: Sequence[Tuple[str, float, float]], filter_order: int, filter_cache: Dict[Tuple[float, float, float], np.ndarray], channel_ok: np.ndarray) -> np.ndarray:
    """Return length ``n_ok_channels * n_bands`` feature vector (NaN where channel not ok)."""
    n_ch = data_ch_x_t.shape[0]
    n_bands = len(bands)
    feats = np.full(n_ch * n_bands, np.nan, dtype=float)
    for bi, (_name, fmin, fmax) in enumerate(bands):
        filtered = GFPBandPower.bandpass_filter_channels(data_ch_x_t, float(fmin), float(fmax), float(sfreq), int(filter_order), filter_cache)
        for ci in range(n_ch):
            if not channel_ok[ci]:
                continue
            row = filtered[ci]
            finite = np.isfinite(row)
            if not np.any(finite):
                continue
            feats[ci * n_bands + bi] = float(np.nanmean(row[finite] ** 2))
        ## END for ci in range(n_ch)...
    ## END for bi, (_name, fmin, fmax) in enumerate(bands)...
    return feats


def _raw_stats_features_for_window(data_ch_x_t: np.ndarray, channel_ok: np.ndarray) -> np.ndarray:
    """Per-channel mean / std / RMS (3 features each); NaN when channel not ok."""
    n_ch = data_ch_x_t.shape[0]
    feats = np.full(n_ch * 3, np.nan, dtype=float)
    for ci in range(n_ch):
        if not channel_ok[ci]:
            continue
        row = data_ch_x_t[ci]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        vals = row[finite]
        feats[ci * 3 + 0] = float(np.mean(vals))
        feats[ci * 3 + 1] = float(np.std(vals)) if vals.size > 1 else 0.0
        feats[ci * 3 + 2] = float(np.sqrt(np.mean(vals ** 2)))
    ## END for ci in range(n_ch)...
    return feats


def build_feature_matrix(eeg_df: pd.DataFrame, channel_names: Sequence[str], sfreq: float, config: EEGManifoldConfig, *, viewport_t_min: Optional[float] = None, viewport_t_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build ``(n_windows, n_features)`` matrix and window-center times from gated EEG."""
    cfg = config.validated()
    empty = (np.zeros((0, 0), dtype=float), np.array([], dtype=float), list(channel_names))
    if eeg_df is None or len(eeg_df) == 0 or "t" not in eeg_df.columns:
        return empty
    cols = [c for c in channel_names if c in eeg_df.columns]
    if len(cols) < cfg.min_channels:
        return empty
    df = eeg_df.sort_values("t").reset_index(drop=True)
    t_vec = df["t"].to_numpy(dtype=float)
    data = df[cols].to_numpy(dtype=float).T
    centers = _window_centers(t_vec, cfg.window_sec, cfg.hop_sec, t_min=viewport_t_min, t_max=viewport_t_max)
    if centers.size == 0:
        return empty
    half = 0.5 * cfg.window_sec
    filter_cache: Dict[Tuple[float, float, float], np.ndarray] = {}
    rows: List[np.ndarray] = []
    kept_centers: List[float] = []
    for center in centers:
        lo, hi = float(center) - half, float(center) + half
        mask = (t_vec >= lo) & (t_vec <= hi)
        if not np.any(mask):
            continue
        win = data[:, mask]
        channel_ok = np.array([_channel_window_ok(win[ci], cfg.max_nan_fraction) for ci in range(win.shape[0])], dtype=bool)
        if int(np.sum(channel_ok)) < cfg.min_channels:
            continue
        if cfg.feature_mode == "raw_stats":
            feat = _raw_stats_features_for_window(win, channel_ok)
        else:
            feat = _band_power_features_for_window(win, float(sfreq), cfg.bands, cfg.filter_order, filter_cache, channel_ok)
        rows.append(feat)
        kept_centers.append(float(center))
    ## END for center in centers...
    if not rows:
        return empty
    return np.vstack(rows), np.asarray(kept_centers, dtype=float), cols


def _standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score with NaN→0 after mean fill; returns ``(Z, means, stds)``."""
    if X.size == 0:
        return X.copy(), np.array([]), np.array([])
    means = np.nanmean(X, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isfinite(X), X, means)
    stds = np.std(filled, axis=0)
    stds = np.where(stds > 1e-12, stds, 1.0)
    Z = (filled - means) / stds
    return Z, means, stds


def _fit_transform_pca(Z: np.ndarray, n_components: int, centers: np.ndarray, fit_history_sec: float, method: ReducerMethod, random_state: int) -> np.ndarray:
    """Rolling or batch PCA / IncrementalPCA → ``(n_windows, n_components)``."""
    from sklearn.decomposition import IncrementalPCA, PCA

    n_win, n_feat = Z.shape
    n_comp = int(min(n_components, n_feat, max(1, n_win - 1)))
    if n_comp < 1 or n_win < 2:
        return np.full((n_win, n_components), np.nan, dtype=float)
    out = np.full((n_win, n_components), np.nan, dtype=float)
    if method == "incremental_pca":
        ipca = IncrementalPCA(n_components=n_comp)
        # Warm-up with first batch large enough for n_comp, then partial_fit walk.
        warm = max(n_comp + 1, min(n_win, max(n_comp * 3, 10)))
        ipca.partial_fit(Z[:warm])
        for i in range(n_win):
            if i >= warm:
                ipca.partial_fit(Z[i : i + 1])
            proj = ipca.transform(Z[i : i + 1])[0]
            out[i, :n_comp] = proj
        ## END for i in range(n_win)...
        return out
    # Rolling / batch PCA
    for i in range(n_win):
        t_i = float(centers[i])
        hist_mask = (centers >= (t_i - fit_history_sec)) & (centers <= t_i)
        idx = np.where(hist_mask)[0]
        if idx.size < max(n_comp + 1, 2):
            idx = np.arange(max(0, i - max(n_comp + 1, 2) + 1), i + 1)
        if idx.size < 2:
            continue
        local_n = int(min(n_comp, idx.size - 1, n_feat))
        if local_n < 1:
            continue
        pca = PCA(n_components=local_n, random_state=random_state)
        pca.fit(Z[idx])
        proj = pca.transform(Z[i : i + 1])[0]
        out[i, :local_n] = proj
    ## END for i in range(n_win)...
    return out


def _fit_transform_umap(Z: np.ndarray, n_components: int, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn is required for method='umap'. Install with: uv add umap-learn") from e
    n_win, n_feat = Z.shape
    n_comp = int(min(n_components, n_feat, max(1, n_win - 1)))
    out = np.full((n_win, n_components), np.nan, dtype=float)
    if n_comp < 1 or n_win < 3:
        return out
    n_neighbors = int(min(max(2, n_neighbors), max(2, n_win - 1)))
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_comp, min_dist=float(min_dist), random_state=int(random_state), metric="euclidean")
    emb = reducer.fit_transform(Z)
    out[:, :n_comp] = np.asarray(emb, dtype=float)
    return out


def embed_feature_matrix(X: np.ndarray, centers: np.ndarray, config: EEGManifoldConfig) -> np.ndarray:
    """Standardize features and embed to ``(n_windows, n_components)``."""
    cfg = config.validated()
    n_win = int(X.shape[0]) if X.ndim == 2 else 0
    out = np.full((n_win, cfg.n_components), np.nan, dtype=float)
    if n_win == 0 or X.size == 0:
        return out
    Z, _means, _stds = _standardize_features(X)
    if cfg.method == "umap":
        return _fit_transform_umap(Z, cfg.n_components, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.random_state)
    method: ReducerMethod = "incremental_pca" if (cfg.live_mode and cfg.method == "pca") else cfg.method
    if method not in ("pca", "incremental_pca"):
        method = "pca"
    return _fit_transform_pca(Z, cfg.n_components, centers, cfg.fit_history_sec, method, cfg.random_state)


def compute_eeg_manifold_from_detailed_df(eeg_df: pd.DataFrame, channel_names: Sequence[str], sfreq: float, *, quality_df: Optional[pd.DataFrame] = None, config: Optional[EEGManifoldConfig] = None, viewport_t_min: Optional[float] = None, viewport_t_max: Optional[float] = None) -> pd.DataFrame:
    """End-to-end manifold computation → DataFrame with ``t`` + ``PC1``…``PCn``."""
    cfg = (config or EEGManifoldConfig()).validated()
    comp_names = cfg.component_names()
    empty = pd.DataFrame({"t": [], **{c: [] for c in comp_names}})
    gated, usable = apply_quality_and_session_gates(eeg_df, quality_df, channel_names, cfg)
    if len(usable) < cfg.min_channels or gated is None or len(gated) == 0:
        return empty
    X, centers, _cols = build_feature_matrix(gated, usable, float(sfreq), cfg, viewport_t_min=viewport_t_min, viewport_t_max=viewport_t_max)
    if centers.size == 0:
        return empty
    emb = embed_feature_matrix(X, centers, cfg)
    out = pd.DataFrame({"t": centers})
    for i, name in enumerate(comp_names):
        out[name] = emb[:, i] if i < emb.shape[1] else np.nan
    ## END for i, name in enumerate(comp_names)...
    return out


def detect_session_bad_channels_from_raws(raws: Sequence[Any], *, skip_trying_PREP: bool = True) -> List[str]:
    """Union of ``time_independent_bad_channels`` across aligned raws."""
    from phopymnehelper.EEG_data import EEGComputations

    bad: set = set()
    for raw in raws:
        if raw is None:
            continue
        try:
            result = EEGComputations.time_independent_bad_channels(raw, skip_trying_PREP=skip_trying_PREP, debug_print=False)
            bad.update(list(result.get("all_bad_channels", []) or []))
        except Exception as e:
            logger.warning("time_independent_bad_channels failed: %s", e)
    ## END for raw in raws...
    return sorted(bad)


__all__ = [
    "DEFAULT_FEATURE_MODE",
    "DEFAULT_FIT_HISTORY_SEC",
    "DEFAULT_HOP_SEC",
    "DEFAULT_MIN_QUALITY_TO_KEEP",
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_WINDOW_SEC",
    "EEGManifoldConfig",
    "FeatureMode",
    "MANIFOLD_COMPONENT_PREFIX",
    "ReducerMethod",
    "apply_quality_and_session_gates",
    "build_feature_matrix",
    "component_column_names",
    "compute_eeg_manifold_from_detailed_df",
    "detect_session_bad_channels_from_raws",
    "embed_feature_matrix",
    "manifold_track_key_for_eeg_name",
    "resolve_usable_channels",
]
