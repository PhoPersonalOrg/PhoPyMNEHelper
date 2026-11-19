"""Result aggregation and multi-session export for batch computations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from phopymnehelper.analysis.computations.batch_executor import BatchSessionResult


class BatchResultsAggregator:
    """Aggregates per-session computation outputs across batch runs into structured DataFrames and summary reports."""

    def __init__(self, batch_results: List[BatchSessionResult]) -> None:
        self.batch_results = batch_results


    def to_dataframe(self, node_id: Optional[str] = None) -> pd.DataFrame:
        """Flatten batch outputs into a single pandas DataFrame.

        If node_id is given, extracts that specific computation output per session.
        Otherwise, flattens all dictionary outputs per session into columns.
        """
        rows: List[Dict[str, Any]] = []

        for res in self.batch_results:
            row: Dict[str, Any] = {
                "session_id": res.session_id,
                "session_path": res.session_path,
                "success": res.success,
                "execution_time_sec": res.execution_time_sec,
            }
            if res.metadata:
                for mk, mv in res.metadata.items():
                    row[f"meta_{mk}"] = mv
                ## END for mk, mv...

            if not res.success:
                row["error"] = str(res.error) if res.error else "Unknown error"
                rows.append(row)
                continue

            if node_id is not None:
                val = res.outputs.get(node_id)
                if isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, (int, float, str, bool, type(None))):
                            row[f"{node_id}_{k}"] = v
                    ## END for k, v...
                elif isinstance(val, (int, float, str, bool)):
                    row[node_id] = val
                elif isinstance(val, pd.DataFrame):
                    row[f"{node_id}_nrows"] = len(val)
            else:
                for nid, val in res.outputs.items():
                    if isinstance(val, dict):
                        for k, v in val.items():
                            if isinstance(v, (int, float, str, bool, type(None))):
                                row[f"{nid}_{k}"] = v
                        ## END for k, v...
                    elif isinstance(val, (int, float, str, bool)):
                        row[nid] = val
                ## END for nid, val...

            rows.append(row)
        ## END for res in self.batch_results...

        return pd.DataFrame(rows)


    def summary_report(self) -> Dict[str, Any]:
        """Generate high-level cohort summary metrics."""
        total = len(self.batch_results)
        successful = sum(1 for r in self.batch_results if r.success)
        failed = total - successful
        total_time = sum(r.execution_time_sec for r in self.batch_results)
        avg_time = (total_time / total) if total > 0 else 0.0

        errors = [
            {"session_id": r.session_id, "error": str(r.error)}
            for r in self.batch_results
            if not r.success and r.error is not None
        ]

        return {
            "total_sessions": total,
            "successful_sessions": successful,
            "failed_sessions": failed,
            "total_execution_time_sec": total_time,
            "avg_execution_time_sec": avg_time,
            "errors": errors,
        }


    def export_csv(self, output_path: Union[str, Path], node_id: Optional[str] = None) -> Path:
        """Export aggregated DataFrame to CSV file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe(node_id=node_id)
        df.to_csv(p, index=False)
        return p


    def export_json(self, output_path: Union[str, Path]) -> Path:
        """Export summary report and result metadata to JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        report = self.summary_report()
        report["sessions"] = [
            {
                "session_id": r.session_id,
                "session_path": r.session_path,
                "success": r.success,
                "execution_time_sec": r.execution_time_sec,
                "output_keys": list(r.outputs.keys()) if r.success else [],
                "error": str(r.error) if r.error else None,
            }
            for r in self.batch_results
        ]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return p


__all__ = [
    "BatchResultsAggregator",
]
