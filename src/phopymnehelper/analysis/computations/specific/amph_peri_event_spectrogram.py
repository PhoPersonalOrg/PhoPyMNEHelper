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
            run=self.run_fn(),
            params_fingerprint=self.amph_peri_event_params_fingerprint,
        )


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
        if "t_unix" in spec and spec["t_unix"] is not None:
            t_unix = np.asarray(spec["t_unix"], dtype=np.float64)
        else:
            t_rel_spec = np.asarray(spec["t"], dtype=np.float64)
            session_t_start = ctx.extras.get("session_t_start_unix")
            if session_t_start is None:
                raise ValueError("amph_peri_event_spectrogram: spectogram has no t_unix and ctx.extras['session_t_start_unix'] is missing")
            t_unix = float(session_t_start) + t_rel_spec


        # -- extract dose event times from ctx.extras or auto-resolve -------
        amph_dose_t_unix = ctx.extras.get("amph_dose_t_unix")
        if amph_dose_t_unix is None:
            if "amph_events" in ctx.extras:
                amph_dose_t_unix = [getattr(ev, "t_unix", ev) for ev in ctx.extras["amph_events"]]
            elif "dose_events" in ctx.extras:
                amph_dose_t_unix = [getattr(ev, "t_unix", ev) for ev in ctx.extras["dose_events"]]
            elif "log_df" in ctx.extras or "text_logger_df" in ctx.extras or hasattr(ctx, "log_df") or hasattr(ctx, "text_logger_df"):
                log_df = ctx.extras.get("log_df", ctx.extras.get("text_logger_df", getattr(ctx, "log_df", getattr(ctx, "text_logger_df", None))))
                if log_df is not None:
                    try:
                        from pypho_timeline.analysis.amph_events import AmphEventParser
                        parsed_events = AmphEventParser.extract_amph_events_from_log_df(log_df)
                        amph_dose_t_unix = [ev.t_unix for ev in parsed_events]
                    except Exception as exc:
                        logger.warning("amph_peri_event_spectrogram: failed to parse log_df for dose events: %s", exc)
                        amph_dose_t_unix = []
                else:
                    amph_dose_t_unix = []
            else:
                logger.info("amph_peri_event_spectrogram: no AMPH dose events found in ctx.extras or log_df; returning 0 peri-event slices")
                amph_dose_t_unix = []
        ## END if amph_dose_t_unix is None...

        amph_dose_t_unix = np.asarray(amph_dose_t_unix, dtype=np.float64)
        if amph_dose_t_unix.size > 0:
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
            freqs=freqs,
            ch_names=ch_names,
            t_rel=(peri_slices[0]["peri"].t_rel if peri_slices else np.asarray([], dtype=np.float64)),
            pre_s=pre_s,
            post_s=post_s,
            computation_version=self.version,
        )


    @classmethod
    def save_result(cls, path: Any, result: Mapping[str, Any], *, event_labels: Optional[Sequence[str]] = None, event_dose_mg: Optional[Sequence[float]] = None) -> Any:
        """Persist amph peri-event result to ``path`` (``.npz``).

        Stacks per-event ``Sxx`` and band curves; stores shared axes and params.
        """
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        peri_slices = list(result.get("peri_slices") or [])
        n_events = int(result.get("n_events") or len(peri_slices))
        freqs = np.asarray(result.get("freqs") if result.get("freqs") is not None else (peri_slices[0]["peri"].freqs if peri_slices else []), dtype=np.float64)
        ch_names = list(result.get("ch_names") or (peri_slices[0]["peri"].ch_names if peri_slices else []))
        t_rel = np.asarray(result.get("t_rel") if result.get("t_rel") is not None else (peri_slices[0]["peri"].t_rel if peri_slices else []), dtype=np.float64)
        pre_s = float(result.get("pre_s", DEFAULT_PERI_PRE_S))
        post_s = float(result.get("post_s", DEFAULT_PERI_POST_S))
        version = str(result.get("computation_version", cls.version))

        payload: Dict[str, Any] = dict(
            n_events=np.asarray(n_events, dtype=np.int64),
            freqs=freqs,
            t_rel=t_rel,
            ch_names=np.asarray(ch_names, dtype=object),
            pre_s=np.asarray(pre_s, dtype=np.float64),
            post_s=np.asarray(post_s, dtype=np.float64),
            computation_version=np.asarray(version),
        )
        if peri_slices:
            payload["event_t_unix"] = np.asarray([float(s["event_t_unix"]) for s in peri_slices], dtype=np.float64)
            payload["coverage_pre_s"] = np.asarray([float(s["coverage_pre_s"]) for s in peri_slices], dtype=np.float64)
            payload["coverage_post_s"] = np.asarray([float(s["coverage_post_s"]) for s in peri_slices], dtype=np.float64)
            payload["Sxx"] = np.stack([np.asarray(s["peri"].Sxx, dtype=np.float32) for s in peri_slices], axis=0)
            band_names = sorted(peri_slices[0]["bands"].keys()) if peri_slices[0].get("bands") else []
            payload["band_names"] = np.asarray(band_names, dtype=object)
            for bname in band_names:
                payload[f"bands_{bname}"] = np.stack([np.asarray(s["bands"][bname], dtype=np.float64) for s in peri_slices], axis=0)
            ## END for bname in band_names....
        else:
            payload["event_t_unix"] = np.asarray([], dtype=np.float64)
            payload["coverage_pre_s"] = np.asarray([], dtype=np.float64)
            payload["coverage_post_s"] = np.asarray([], dtype=np.float64)
            payload["band_names"] = np.asarray([], dtype=object)

        if event_labels is not None:
            payload["label_raw"] = np.asarray(list(event_labels), dtype=object)
        if event_dose_mg is not None:
            payload["dose_mg"] = np.asarray(list(event_dose_mg), dtype=np.float64)

        sxx_pre = result.get("Sxx_pre_mean")
        sxx_post = result.get("Sxx_post_mean")
        if sxx_pre is not None and np.asarray(sxx_pre).size > 0:
            payload["Sxx_pre_mean"] = np.asarray(sxx_pre, dtype=np.float32)
        if sxx_post is not None and np.asarray(sxx_post).size > 0:
            payload["Sxx_post_mean"] = np.asarray(sxx_post, dtype=np.float32)

        np.savez_compressed(out, **payload)
        return out


    @classmethod
    def load_result(cls, path: Any) -> Dict[str, Any]:
        """Load amph peri-event NPZ written by :meth:`save_result` into a plain dict."""
        from pathlib import Path

        data = np.load(Path(path), allow_pickle=True)
        n_events = int(np.asarray(data["n_events"]).reshape(-1)[0])
        freqs = np.asarray(data["freqs"], dtype=np.float64)
        t_rel = np.asarray(data["t_rel"], dtype=np.float64)
        ch_names = [str(c) for c in np.asarray(data["ch_names"]).tolist()] if "ch_names" in data.files else []
        band_names = [str(b) for b in np.asarray(data["band_names"]).tolist()] if "band_names" in data.files else []
        event_t_unix = np.asarray(data["event_t_unix"], dtype=np.float64) if "event_t_unix" in data.files else np.asarray([], dtype=np.float64)
        coverage_pre_s = np.asarray(data["coverage_pre_s"], dtype=np.float64) if "coverage_pre_s" in data.files else np.asarray([], dtype=np.float64)
        coverage_post_s = np.asarray(data["coverage_post_s"], dtype=np.float64) if "coverage_post_s" in data.files else np.asarray([], dtype=np.float64)
        Sxx_stack = np.asarray(data["Sxx"], dtype=np.float32) if "Sxx" in data.files else None
        bands_by_name = {b: np.asarray(data[f"bands_{b}"], dtype=np.float64) for b in band_names if f"bands_{b}" in data.files}

        peri_slices: List[Dict[str, Any]] = []
        if Sxx_stack is not None and n_events > 0:
            for i in range(n_events):
                bands_i = {b: bands_by_name[b][i] for b in band_names if b in bands_by_name}
                peri_slices.append(dict(
                    event_t_unix=float(event_t_unix[i]) if i < event_t_unix.size else float("nan"),
                    coverage_pre_s=float(coverage_pre_s[i]) if i < coverage_pre_s.size else 0.0,
                    coverage_post_s=float(coverage_post_s[i]) if i < coverage_post_s.size else 0.0,
                    Sxx=Sxx_stack[i],
                    bands=bands_i,
                    t_rel=t_rel,
                    freqs=freqs,
                    ch_names=ch_names,
                ))
            ## END for i in range(n_events)....

        out: Dict[str, Any] = dict(
            peri_slices=peri_slices,
            n_events=n_events,
            freqs=freqs,
            t_rel=t_rel,
            ch_names=ch_names,
            pre_s=float(np.asarray(data["pre_s"]).reshape(-1)[0]) if "pre_s" in data.files else DEFAULT_PERI_PRE_S,
            post_s=float(np.asarray(data["post_s"]).reshape(-1)[0]) if "post_s" in data.files else DEFAULT_PERI_POST_S,
            computation_version=str(np.asarray(data["computation_version"]).reshape(-1)[0]) if "computation_version" in data.files else cls.version,
            event_t_unix=event_t_unix,
            coverage_pre_s=coverage_pre_s,
            coverage_post_s=coverage_post_s,
            Sxx=Sxx_stack,
            bands=bands_by_name,
        )
        if "label_raw" in data.files:
            out["label_raw"] = [str(x) for x in np.asarray(data["label_raw"]).tolist()]
        if "dose_mg" in data.files:
            out["dose_mg"] = np.asarray(data["dose_mg"], dtype=np.float64)
        if "Sxx_pre_mean" in data.files:
            out["Sxx_pre_mean"] = np.asarray(data["Sxx_pre_mean"], dtype=np.float32)
        if "Sxx_post_mean" in data.files:
            out["Sxx_post_mean"] = np.asarray(data["Sxx_post_mean"], dtype=np.float32)
        return out


__all__ = [
    "AMPH_PERI_EVENT_PARAM_KEYS",
    "AmphPeriEventSpectrogramComputation",
    "DEFAULT_PERI_PRE_S",
    "DEFAULT_PERI_POST_S",
    "DEFAULT_BANDS",
]
