"""AMPH dose peri-event spectrogram computation.

Slices full-session EEG spectrograms around AMPH dose events and produces
event-mean pre/post TFRs plus per-event band-power curves.

See ``phopymnehelper/analysis/COMPUTATIONS_README.md`` for the computations contract.

Dependencies
------------
- ``"spectogram"`` node (provides the full-session STFT result dict).
- AMPH dose event timestamps passed via ``ctx.extras["amph_dose_t_unix"]``
  (a 1-D array-like of unix seconds for each dose event).

Output structure (dict)
-----------------------
- ``peri_slices``:  list of per-event dicts, each containing:
    - ``event_t_unix``: float
    - ``peri``: :class:`PeriEventSlice`
    - ``bands``: dict mapping band name -> 1-D array (time)
    - ``coverage_pre_s``, ``coverage_post_s``: floats
- ``Sxx_pre_mean``: (n_ch, n_freqs, n_pre_times) nanmean over events
- ``Sxx_post_mean``: (n_ch, n_freqs, n_post_times) nanmean over events
- ``n_events``: int
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from phopymnehelper.analysis.computations.protocol import ArtifactKind, RunContext
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase

logger = logging.getLogger(__name__)

# Defaults mirroring pyPhoTimeline.analysis.peri_event
DEFAULT_PERI_PRE_S: float = 600.0
DEFAULT_PERI_POST_S: float = 1800.0

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

AMPH_PERI_EVENT_PARAM_KEYS: FrozenSet[str] = frozenset({"pre_s", "post_s", "bands"})


class AmphPeriEventSpectrogramComputation(SpecificComputationBase):
    """Peri-event spectrogram slicing around AMPH dose events, wired as DAG node ``"amph_peri_event_spectrogram"``.

    Depends on ``"spectogram"`` so the full-session STFT is available in ``dep_outputs``.
    AMPH dose event timestamps are supplied via ``ctx.extras["amph_dose_t_unix"]``.

    Parameters (merged from graph params)
    --------------------------------------
    pre_s : float
        Seconds before each dose event to include (default 600).
    post_s : float
        Seconds after each dose event to include (default 1800).
    bands : dict[str, tuple[float, float]]
        Frequency band definitions for band-power curves (default: delta-gamma).
    """

    computation_id: ClassVar[str] = "amph_peri_event_spectrogram"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ("spectogram",)
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.object


    @classmethod
    def filter_params(cls, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Extract only the keys relevant to this computation."""
        return {k: params[k] for k in AMPH_PERI_EVENT_PARAM_KEYS if k in params}


    @classmethod
    def amph_peri_event_params_fingerprint(cls, params: Mapping[str, Any]) -> str:
        """Stable JSON fingerprint of peri-event-specific params only."""
        f = cls.filter_params(params)
        return json.dumps({k: f[k] for k in sorted(f.keys())}, sort_keys=True, default=str)


    def to_computation_node(self) -> Any:
        from phopymnehelper.analysis.computations.protocol import ComputationNode
        return ComputationNode(
            id=self.computation_id,
            version=self.version,
            deps=self.deps,
            kind=self.artifact_kind,
            run=self.run,
            params_fingerprint=self.amph_peri_event_params_fingerprint,
        )


    def run(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        """Thin wrapper delegating to :meth:`compute` with lifecycle hooks."""
        return self._run_with_hooks(ctx, params, dep_outputs)


    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Dict[str, Any]:
        """Slice peri-event spectrograms around each AMPH dose event and average.

        Expects
        -------
        dep_outputs["spectogram"]
            Dict with keys ``Sxx`` (n_ch, n_freqs, n_times), ``freqs``, ``t_unix``, ``ch_names``
            (or a structure whose values can be coerced to arrays).
        ctx.extras["amph_dose_t_unix"]
            1-D array-like of unix timestamps for dose events.
        """
        from phopymnehelper.analysis.computations.peri_event_helpers import PeriEventHelpers

        # -- extract spectrogram from upstream node --------------------------
        spec = dep_outputs.get("spectogram")
        if spec is None:
            raise ValueError("amph_peri_event_spectrogram: upstream 'spectogram' output is None")

        Sxx = spec["Sxx"]
        if hasattr(Sxx, "values"):
            Sxx = Sxx.values
        Sxx = np.asarray(Sxx, dtype=np.float32)
        freqs = np.asarray(spec["freqs"], dtype=np.float64)
        ch_names: List[str] = list(spec["ch_names"])

        # t_unix: full-session spectrogram time axis in unix seconds
        t_unix = np.asarray(spec["t_unix"], dtype=np.float64)

        # -- extract dose event times from ctx.extras -----------------------
        amph_dose_t_unix = ctx.extras.get("amph_dose_t_unix")
        if amph_dose_t_unix is None:
            raise ValueError("amph_peri_event_spectrogram: ctx.extras['amph_dose_t_unix'] not provided")
        amph_dose_t_unix = np.asarray(amph_dose_t_unix, dtype=np.float64)
        amph_dose_t_unix = np.unique(amph_dose_t_unix)  # drop duplicate doses

        # -- params ----------------------------------------------------------
        kw = self.filter_params(params)
        pre_s: float = float(kw.get("pre_s", DEFAULT_PERI_PRE_S))
        post_s: float = float(kw.get("post_s", DEFAULT_PERI_POST_S))
        bands = kw.get("bands", DEFAULT_BANDS)

        # -- filter dose events to those inside spectrogram span ------------
        if t_unix.size > 0:
            in_window = (amph_dose_t_unix >= float(t_unix.min())) & (amph_dose_t_unix <= float(t_unix.max()))
            session_dose_ts = amph_dose_t_unix[in_window]
        else:
            session_dose_ts = np.array([], dtype=np.float64)

        # -- per-event slicing -----------------------------------------------
        peri_slices: List[Dict[str, Any]] = []
        Sxx_pre_list: List[np.ndarray] = []
        Sxx_post_list: List[np.ndarray] = []

        for event_t in session_dose_ts:
            a_peri = PeriEventHelpers.slice_peri_event_spectrogram(
                Sxx=Sxx,
                freqs=freqs,
                t_unix=t_unix,
                ch_names=ch_names,
                event_t_unix=float(event_t),
                pre_s=pre_s,
                post_s=post_s,
            )
            sxx_2d = PeriEventHelpers.mean_sxx_over_channels(a_peri.Sxx)
            a_bands = PeriEventHelpers.band_power_curves(sxx_2d, a_peri.freqs, bands=bands)
            peri_slices.append(dict(
                event_t_unix=float(event_t),
                peri=a_peri,
                bands=a_bands,
                coverage_pre_s=a_peri.coverage_pre_s,
                coverage_post_s=a_peri.coverage_post_s,
            ))

            pre_mask = a_peri.t_rel < 0
            post_mask = a_peri.t_rel > 0
            # spectrogram (n_ch, n_f, n_t)
            Sxx_pre_list.append(a_peri.Sxx[:, :, pre_mask])
            Sxx_post_list.append(a_peri.Sxx[:, :, post_mask])
        ## END for event_t in session_dose_ts...

        # -- event-mean pre/post spectrograms --------------------------------
        if Sxx_pre_list:
            Sxx_pre_mean = np.nanmean(Sxx_pre_list, axis=0)
        else:
            Sxx_pre_mean = np.array([])

        if Sxx_post_list:
            Sxx_post_mean = np.nanmean(Sxx_post_list, axis=0)
        else:
            Sxx_post_mean = np.array([])

        n_events = len(peri_slices)
        logger.info("amph_peri_event_spectrogram: %d dose event(s) sliced (of %d total)", n_events, amph_dose_t_unix.size)

        return dict(
            peri_slices=peri_slices,
            Sxx_pre_mean=Sxx_pre_mean,
            Sxx_post_mean=Sxx_post_mean,
            n_events=n_events,
        )


__all__ = [
    "AMPH_PERI_EVENT_PARAM_KEYS",
    "AmphPeriEventSpectrogramComputation",
    "DEFAULT_PERI_PRE_S",
    "DEFAULT_PERI_POST_S",
]
