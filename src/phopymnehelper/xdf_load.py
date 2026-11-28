"""Resilient ``pyxdf.load_xdf`` wrapper that quiets recoverable truncated-chunk spam."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pyxdf

PathLike = Union[str, Path]

_logger = logging.getLogger(__name__)


class _TruncationFlagHandler(logging.Handler):
    """Captures recoverable pyxdf truncation messages without printing them."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.saw_truncation: bool = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(getattr(record, "msg", ""))
        if ("Error reading chunk length" in msg) or ("zero-length chunk" in msg) or ("invalid variable-length integer" in msg) or ("scanning forward to next" in msg):
            self.saw_truncation = True


@contextmanager
def quiet_pyxdf_recovery_logs():
    """Temporarily silence pyxdf loggers while capturing truncation recovery flags."""
    probe = _TruncationFlagHandler()
    targets = [logging.getLogger("pyxdf"), logging.getLogger("pyxdf.pyxdf")]
    prev: List[Tuple[logging.Logger, int, bool, List[logging.Handler]]] = []
    for lg in targets:
        prev.append((lg, lg.level, lg.propagate, list(lg.handlers)))
        lg.handlers = [probe]
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
    ## END for lg in targets...

    try:
        yield probe
    finally:
        for lg, level, propagate, handlers in prev:
            lg.handlers = handlers
            lg.setLevel(level)
            lg.propagate = propagate
        ## END for lg, level, propagate, handlers in prev...


def load_xdf_resilient(filename: PathLike, *, synchronize_clocks: bool = True, handle_clock_resets: bool = True, dejitter_timestamps: bool = False, verbose: bool = False, warn: bool = True, **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Call ``pyxdf.load_xdf`` with recoverable truncated-chunk traceback spam suppressed.

    When pyxdf encounters a truncated/corrupt trailing chunk it recovers by scanning for the
    next boundary. Upstream ``logger.exception`` dumps a full traceback that looks like a crash;
    this wrapper keeps that quiet and emits a single concise warning naming the file instead.
    """
    path = Path(filename)
    with quiet_pyxdf_recovery_logs() as probe:
        streams, header = pyxdf.load_xdf(str(path), synchronize_clocks=synchronize_clocks, handle_clock_resets=handle_clock_resets, dejitter_timestamps=dejitter_timestamps, verbose=verbose, **kwargs)
    if warn and probe.saw_truncation:
        _logger.warning("XDF appears truncated/corrupt at end (pyxdf recovered remaining chunks): %s", path.name)
    return streams, header


__all__ = ["load_xdf_resilient", "quiet_pyxdf_recovery_logs"]
