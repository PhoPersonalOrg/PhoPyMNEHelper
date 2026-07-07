from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from phopylslhelper.datetime_helpers import DISPLAY_TIMEZONE, datetime_to_unix_timestamp

logger = logging.getLogger(__name__)

_IOGRAPH_FILENAME_TIME_RANGE_RE = re.compile(r"\(from\s+(\d{1,2})-(\d{2})\s+to\s+(\d{1,2})-(\d{2})\)", re.IGNORECASE)
_IOGRAPH_FILENAME_TIME_RANGE_DATED_RE = re.compile(r"\(from\s+(\d{1,2})-(\d{2})\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\s+to\s+(\d{1,2})-(\d{2})\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\)", re.IGNORECASE)


class IOGraphProcessor:
    """ 

    """
    all_psychometric_detail_cols = ["backtrack_severity", "impairment_metric", "grab_failed_event"]


    def __init__(self, *, directory: Optional[Path] = None, file_table_df: Optional[pd.DataFrame] = None, master_df: Optional[pd.DataFrame] = None, recursive: bool = False, drop_na_coords: bool = True):
        self.directory = Path(directory).resolve() if directory is not None else None
        self.recursive = bool(recursive)
        self.drop_na_coords = bool(drop_na_coords)
        self.file_table_df = file_table_df.copy() if file_table_df is not None else pd.DataFrame()
        if master_df is None and (not self.file_table_df.empty):
            computed_master_df, _ = self.build_master_df(self.file_table_df, output_csv=None, drop_na_coords=self.drop_na_coords)
            self.master_df = computed_master_df
        elif master_df is not None:
            self.master_df = master_df.copy()
        else:
            self.master_df = pd.DataFrame(
                columns=[
                    "session_index",
                    "sample_time",
                    "time",
                    "x",
                    "y",
                    "source_file_name",
                    "parsed_filename_dt_start",
                    "parsed_filename_dt_end",
                ]
            )


    @classmethod
    def from_directory(cls, directory: Path, *, recursive: bool = False, drop_na_coords: bool = True) -> "IOGraphProcessor":
        file_table_df, _ = cls.scan_csv_directory(directory, output_csv=None, recursive=recursive)
        master_df, _ = cls.build_master_df(file_table_df, output_csv=None, drop_na_coords=drop_na_coords)
        return cls(
            directory=directory,
            file_table_df=file_table_df,
            master_df=master_df,
            recursive=recursive,
            drop_na_coords=drop_na_coords,
        )

    @property
    def detail_df(self) -> pd.DataFrame:
        return self.master_df_to_detail_df(self.master_df)

    @property
    def intervals_df(self) -> pd.DataFrame:
        return self.master_df_to_intervals_df(self.master_df)

    def save_file_table_csv(self, output_csv: Path) -> Path:
        out_path = Path(output_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_table_df.to_csv(out_path, index=False)
        return out_path

    def save_master_csv(self, output_csv: Path) -> Path:
        out_path = Path(output_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.master_df.to_csv(out_path, index=False)
        return out_path

    @classmethod
    def _parsed_filename_dt_range(cls, file_name: str, modified_datetime: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        def _subfn_parse_iograph_day_label(day_label: str, year: int) -> pd.Timestamp:
            cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", day_label.strip(), flags=re.IGNORECASE)
            return pd.to_datetime(f"{cleaned} {year}", format="%b %d %Y")

        if pd.isna(modified_datetime):
            return pd.NaT, pd.NaT
        year = pd.Timestamp(modified_datetime).year
        dated_match = _IOGRAPH_FILENAME_TIME_RANGE_DATED_RE.search(file_name)
        if dated_match is not None:
            start_h, start_m, start_day, end_h, end_m, end_day = dated_match.groups()
            day_start, day_end = _subfn_parse_iograph_day_label(start_day, year), _subfn_parse_iograph_day_label(end_day, year)
            if day_end.normalize() < day_start.normalize():
                day_end = _subfn_parse_iograph_day_label(end_day, year + 1)
            dt_start = day_start + pd.Timedelta(hours=int(start_h), minutes=int(start_m))
            dt_end = day_end + pd.Timedelta(hours=int(end_h), minutes=int(end_m))
            return dt_start, dt_end
        match = _IOGRAPH_FILENAME_TIME_RANGE_RE.search(file_name)
        if match is None:
            return pd.NaT, pd.NaT
        start_h, start_m, end_h, end_m = (int(match.group(i)) for i in range(1, 5))
        base_date = pd.Timestamp(modified_datetime).normalize()
        dt_start = base_date + pd.Timedelta(hours=start_h, minutes=start_m)
        dt_end = base_date + pd.Timedelta(hours=end_h, minutes=end_m)
        if dt_end < dt_start:
            dt_end += pd.Timedelta(days=1)
        return dt_start, dt_end


    @classmethod
    def scan_csv_directory(cls, directory: Path, output_csv: Path | None = None, *, recursive: bool = False) -> tuple[pd.DataFrame, Path | None]:
        root, out_path = Path(directory).resolve(), (Path(output_csv).resolve() if output_csv is not None else None)
        pattern = "**/*.csv" if recursive else "*.csv"
        records = []
        for p in sorted(root.glob(pattern)):
            if not p.is_file() or (out_path is not None and p.resolve() == out_path):
                continue
            st = p.stat()
            modified_local = pd.to_datetime(st.st_mtime, unit="s", utc=True).tz_convert(DISPLAY_TIMEZONE).tz_localize(None)
            created_local = pd.to_datetime(st.st_ctime, unit="s", utc=True).tz_convert(DISPLAY_TIMEZONE).tz_localize(None)
            records.append({"file_path": str(p.resolve()), "file_name": p.name, "created_datetime": created_local, "modified_datetime": modified_local, "size_bytes": st.st_size})
        results_df = pd.DataFrame(records)
        if not results_df.empty:
            parsed = results_df.apply(lambda row: cls._parsed_filename_dt_range(row["file_name"], row["modified_datetime"]), axis=1, result_type="expand")
            parsed.columns = ["parsed_filename_dt_start", "parsed_filename_dt_end"]
            results_df = pd.concat([results_df, parsed], axis=1)
        else:
            results_df["parsed_filename_dt_start"] = pd.Series(dtype="datetime64[ns]")
            results_df["parsed_filename_dt_end"] = pd.Series(dtype="datetime64[ns]")
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out_path, index=False)
        return results_df, out_path


    @classmethod
    def build_master_df(cls, file_table_df: pd.DataFrame, output_csv: Path | None = None, *, drop_na_coords: bool = True) -> tuple[pd.DataFrame, Path | None]:
        """ build the complete dataframe with all non-duplicate entries
        """
        def _subfn_intervals_overlap(start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp) -> bool:
            return start_a < end_b and start_b < end_a

        def _subfn_assign_session_groups(session_table: pd.DataFrame) -> pd.DataFrame:
            """Merge files whose filename intervals overlap; rank by earliest snapshot for dedupe.

            IOGraph sessions are often saved incrementally (same start, growing end). Overlapping
            sample_time rows keep data from the earliest/shorter export.
            """

            n = len(session_table)
            parent = list(range(n))

            def find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def union(i: int, j: int) -> None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

            starts = session_table["parsed_filename_dt_start"].values
            ends = session_table["parsed_filename_dt_end"].values
            for i in range(n):
                for j in range(i + 1, n):
                    if _subfn_intervals_overlap(starts[i], ends[i], starts[j], ends[j]):
                        union(i, j)
            session_table = session_table.copy()
            session_table["_merge_root"] = [find(i) for i in range(n)]
            group_starts = session_table.groupby("_merge_root")["parsed_filename_dt_start"].min()
            root_to_session = {root: idx for idx, root in enumerate(group_starts.sort_values(kind="stable").index)}
            session_table["session_index"] = session_table["_merge_root"].map(root_to_session)
            rank_cols = ["parsed_filename_dt_end", "parsed_filename_dt_start", "modified_datetime", "size_bytes"]
            for col in rank_cols:
                if col not in session_table.columns:
                    session_table[col] = pd.NaT if col != "size_bytes" else 0
            session_table = session_table.sort_values(["session_index"] + rank_cols, kind="stable")
            session_table["file_rank_in_session"] = session_table.groupby("session_index", sort=False).cumcount()
            for sess_idx, grp in session_table.groupby("session_index", sort=True):
                if len(grp) > 1:
                    ordered = grp.sort_values(rank_cols, kind="stable")
                    names = ordered["file_name"].tolist() if "file_name" in ordered.columns else [Path(p).name for p in ordered["file_path"]]
                    logger.info("Merged %s files into session %s: %s", len(grp), sess_idx, ", ".join(names))
            return session_table.drop(columns=["_merge_root"])

        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        required_cols = ["file_path", "parsed_filename_dt_start", "parsed_filename_dt_end"]
        missing_cols = [c for c in required_cols if c not in file_table_df.columns]
        if missing_cols:
            raise ValueError(f"file_table_df missing required columns: {missing_cols}")
        session_table = file_table_df.dropna(subset=["parsed_filename_dt_end"]).copy()
        session_table["parsed_filename_dt_start"] = pd.to_datetime(session_table["parsed_filename_dt_start"])
        session_table["parsed_filename_dt_end"] = pd.to_datetime(session_table["parsed_filename_dt_end"])
        session_table = session_table.sort_values("parsed_filename_dt_start", kind="stable").reset_index(drop=True)
        session_table = _subfn_assign_session_groups(session_table)
        parts: list[pd.DataFrame] = []
        skipped_empty = 0
        for _, row in session_table.iterrows():
            df = pd.read_csv(row["file_path"])
            if drop_na_coords:
                df = df.dropna(subset=["x", "y"])
            if df.empty:
                skipped_empty += 1
                continue
            t_max = df["time"].max()
            dt_end = pd.Timestamp(row["parsed_filename_dt_end"])
            df["sample_time"] = dt_end - pd.to_timedelta(t_max - df["time"], unit="ms")
            df["session_index"] = int(row["session_index"])
            df["file_rank_in_session"] = int(row["file_rank_in_session"])
            df["source_file_name"] = row.get("file_name", Path(row["file_path"]).name)
            df["parsed_filename_dt_start"] = row["parsed_filename_dt_start"]
            df["parsed_filename_dt_end"] = row["parsed_filename_dt_end"]
            parts.append(df[["session_index", "file_rank_in_session", "sample_time", "time", "x", "y", "source_file_name", "parsed_filename_dt_start", "parsed_filename_dt_end"]])
        if skipped_empty:
            logger.info("Skipped %s session(s) with no valid coordinates after drop_na_coords.", skipped_empty)
        master_cols = ["session_index", "sample_time", "time", "x", "y", "source_file_name", "parsed_filename_dt_start", "parsed_filename_dt_end"]
        master_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=master_cols)
        if not master_df.empty:
            master_df = master_df.sort_values(["session_index", "file_rank_in_session", "sample_time"], kind="stable")
            master_df = master_df.drop_duplicates(subset=["session_index", "sample_time"], keep="first")
            master_df = master_df.drop(columns=["file_rank_in_session"]).sort_values("sample_time", kind="stable").reset_index(drop=True)
        out_path = Path(output_csv).resolve() if output_csv is not None else None
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            master_df.to_csv(out_path, index=False)
        return master_df, out_path


    @classmethod
    def master_df_to_detail_df(cls, master_df: pd.DataFrame) -> pd.DataFrame:
        if master_df.empty:
            return pd.DataFrame(columns=["t", "x", "y", "session_index", "source_file_name"])
        detail_df = master_df[["sample_time", "x", "y", "session_index", "source_file_name"]].copy()
        detail_df["t"] = detail_df["sample_time"].apply(datetime_to_unix_timestamp)
        detail_df = detail_df.drop(columns=["sample_time"]).sort_values("t", kind="stable").reset_index(drop=True)
        return detail_df[["t", "x", "y", "session_index", "source_file_name"]]


    def computing_psychomotor_metric_columns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute psychomotor metrics from a detail dataframe (`t`, `x`, `y`).

        new_cols_df, intervals_df = IOGraphProcessor.compute_psychomotor_metrics(detail_df=self.detail_df)

        Vectorized implementation: prefix sums plus ``np.searchsorted`` replace the
        original nested Python loops so cost is ~O(n log n) instead of O(n * window),
        with no per-row ``df.loc`` scalar access. Behavior matches the original for a
        ``t``-sorted ``detail_df`` (as produced by ``master_df_to_detail_df``).
        """
        detail_cols = self.all_psychometric_detail_cols # ["backtrack_severity", "impairment_metric", "grab_failed_event"]

        new_cols_df, intervals_df = self.compute_psychomotor_metrics(detail_df=self.detail_df)
        ## add the columns to self
        assert len(self.master_df) == len(new_cols_df), f"len(self.master_df): {len(self.master_df)} != len(new_cols_df): {len(new_cols_df)}"
        self.master_df[detail_cols] = new_cols_df[detail_cols] ## add new columns to master_df


    @classmethod
    def compute_psychomotor_metrics(cls, detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute psychomotor metrics from a detail dataframe (`t`, `x`, `y`).

        new_cols_df, intervals_df = IOGraphProcessor.compute_psychomotor_metrics(detail_df=self.detail_df)

        Vectorized implementation: prefix sums plus ``np.searchsorted`` replace the
        original nested Python loops so cost is ~O(n log n) instead of O(n * window),
        with no per-row ``df.loc`` scalar access. Behavior matches the original for a
        ``t``-sorted ``detail_df`` (as produced by ``master_df_to_detail_df``).
        """
        detail_cols = ["t", *cls.all_psychometric_detail_cols] # ["backtrack_severity", "impairment_metric", "grab_failed_event"]
        interval_cols = ["t_start", "t_duration", *cls.all_psychometric_detail_cols] # ["backtrack_severity", "impairment_metric", "grab_failed_event"]
        if detail_df.empty or len(detail_df) < 2:
            return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=interval_cols)

        t = detail_df["t"].to_numpy(dtype=float)
        x = detail_df["x"].to_numpy(dtype=float)
        y = detail_df["y"].to_numpy(dtype=float)
        n = t.shape[0]

        # Per-step deltas (index 0 == 0.0, matching the original `.diff().fillna(0)`).
        dt = np.diff(t, prepend=t[0])
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        angle_diff = np.diff(angle, prepend=angle[0])
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        abs_adiff = np.abs(angle_diff)

        # Prefix sums over [0, k): sum(arr[a..b]) inclusive == prefix[b + 1] - prefix[a].
        # `dt >= 0` for t-sorted input keeps `p_dt` monotone so searchsorted is valid.
        p_dt = np.concatenate(([0.0], np.cumsum(dt)))
        p_dist = np.concatenate(([0.0], np.cumsum(dist)))
        p_grab = np.concatenate(([0], np.cumsum(abs_adiff > (np.pi / 2))))

        # Trigger indices: a "backtrack" is a sharp reversal; outer loop starts at i=1.
        triggers = np.nonzero(abs_adiff > (3 * np.pi / 4))[0]
        triggers = triggers[triggers >= 1]

        bs = np.zeros(n)
        im = np.zeros(n)
        gf = np.zeros(n)

        if triggers.size == 0:
            result_df = pd.DataFrame(
                {"t": t, "backtrack_severity": bs, "impairment_metric": im, "grab_failed_event": gf},
                index=detail_df.index,
            )
            return result_df, pd.DataFrame(columns=interval_cols)

        # Forward window per trigger: largest q with sum(dt[i..q-1]) <= 1.0.
        q_max = np.searchsorted(p_dt, p_dt[triggers] + 1.0, side="right") - 1
        backtrack_dur = p_dt[q_max] - p_dt[triggers]
        backtrack_dist = p_dist[q_max] - p_dist[triggers]
        end_idx = np.minimum(q_max, n - 1)

        # grab_fail_count: reversals > pi/2 among window indices (i, q_max - 1].
        gi1 = np.minimum(triggers + 1, n)
        grab_count = np.where(q_max >= triggers + 1, p_grab[q_max] - p_grab[gi1], 0)

        # Trajectory accumulated over non-trigger steps since the previous trigger
        # (index 0 acts as the initial boundary), which resets at every trigger.
        prev = np.empty_like(triggers)
        prev[0] = 0
        prev[1:] = triggers[:-1]
        traj_dist = p_dist[triggers] - p_dist[prev + 1]
        traj_dur = p_dt[triggers] - p_dt[prev + 1]

        recorded = (
            (backtrack_dur > 0)
            & (backtrack_dur < 1.0)
            & (traj_dist > 0)
            & (traj_dur > 0)
            & (backtrack_dist < traj_dist)
        )

        severity = np.zeros(triggers.shape[0])
        np.divide(backtrack_dist, traj_dist, out=severity, where=recorded)
        severity *= 100.0
        impairment = np.minimum(severity / 100.0, 1.0)
        grab_failed = np.where(grab_count >= 2, 1.0, 0.0)

        rec_pos = np.nonzero(recorded)[0]
        # Fill per-row columns in ascending trigger order so overlapping windows keep
        # the original "last writer wins" semantics.
        for k in rec_pos:
            i0 = triggers[k]
            e = end_idx[k] + 1
            bs[i0:e] = severity[k]
            im[i0:e] = impairment[k]
            gf[i0:e] = grab_failed[k]
        ## END for k in rec_pos...

        result_df = pd.DataFrame(
            {"t": t, "backtrack_severity": bs, "impairment_metric": im, "grab_failed_event": gf},
            index=detail_df.index,
        )

        if rec_pos.size == 0:
            intervals_df = pd.DataFrame(columns=interval_cols)
        else:
            intervals_df = pd.DataFrame(
                {
                    "t_start": t[triggers[rec_pos]],
                    "t_duration": backtrack_dur[rec_pos],
                    "backtrack_severity": severity[rec_pos],
                    "impairment_metric": impairment[rec_pos],
                    "grab_failed_event": grab_failed[rec_pos],
                }
            )

        return result_df, intervals_df


    @classmethod
    def master_df_to_intervals_df(cls, master_df: pd.DataFrame) -> pd.DataFrame:
        present_psychometric_cols = [c for c in cls.all_psychometric_detail_cols if c in master_df.columns]
        if master_df.empty:
            return pd.DataFrame(columns=["t_start", "t_duration", "session_index", "source_file_names", "parsed_filename_dt_start", "parsed_filename_dt_end", *present_psychometric_cols])
        import polars as pl
        df = pl.from_pandas(master_df)

        # Convert any potential string columns to datetime first to ensure safety
        for col_name in ["parsed_filename_dt_start", "parsed_filename_dt_end"]:
            if df.schema[col_name].base_type() != pl.Datetime:
                try:
                    df = df.with_columns(pl.col(col_name).str.to_datetime())
                except Exception:
                    try:
                        df = df.with_columns(pl.col(col_name).cast(pl.Datetime))
                    except Exception:
                        pass

        out = (
            df.group_by("session_index", maintain_order=False)
            .agg(
                pl.col("parsed_filename_dt_start").min().alias("parsed_filename_dt_start"),
                pl.col("parsed_filename_dt_end").max().alias("parsed_filename_dt_end"),
                pl.col("source_file_name").unique().sort().str.join(", ").alias("source_file_names"),
                *[pl.col(c).max().alias(c) for c in present_psychometric_cols]
            )
            .sort("session_index")
            .with_columns(
                # Use microseconds then divide to float seconds to match float(datetime_to_unix_timestamp)
                (pl.col("parsed_filename_dt_start").cast(pl.Datetime).dt.timestamp("us") / 1000000.0).alias("t_start"),
                (pl.col("parsed_filename_dt_end").cast(pl.Datetime).dt.timestamp("us") / 1000000.0).alias("t_end")
            )
            .with_columns(
                (pl.col("t_end") - pl.col("t_start")).alias("t_duration")
            )
            .select(
                "t_start",
                "t_duration",
                "session_index",
                "source_file_names",
                "parsed_filename_dt_start",
                "parsed_filename_dt_end",
                *present_psychometric_cols
            )
        )
        # Ensure proper output type for psychometric columns if any (converting them to float)
        out_pandas = out.to_pandas()
        for col in present_psychometric_cols:
            out_pandas[col] = out_pandas[col].astype(float)
        return out_pandas


__all__ = ["IOGraphProcessor"]
