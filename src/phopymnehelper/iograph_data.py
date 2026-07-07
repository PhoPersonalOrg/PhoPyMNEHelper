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
    def __init__(
        self,
        *,
        directory: Optional[Path] = None,
        file_table_df: Optional[pd.DataFrame] = None,
        master_df: Optional[pd.DataFrame] = None,
        recursive: bool = False,
        drop_na_coords: bool = True,
    ):
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


    @classmethod
    def compute_psychomotor_metrics(cls, detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute psychomotor metrics from a detail dataframe (`t`, `x`, `y`).

        new_cols_df, intervals_df = IOGraphProcessor.compute_psychomotor_metrics(detail_df=self.detail_df)

        """
        if detail_df.empty or len(detail_df) < 2:
            return (
                pd.DataFrame(columns=["t", "backtrack_severity", "impairment_metric", "grab_failed_event"]),
                pd.DataFrame(columns=["t_start", "t_duration", "backtrack_severity", "impairment_metric", "grab_failed_event"]),
            )

        df = detail_df.copy()
        df["dt"] = df["t"].diff().fillna(0)
        df["dx"] = df["x"].diff().fillna(0)
        df["dy"] = df["y"].diff().fillna(0)
        df["angle"] = np.arctan2(df["dy"], df["dx"])
        df["angle_diff"] = df["angle"].diff().fillna(0)
        df["angle_diff"] = (df["angle_diff"] + np.pi) % (2 * np.pi) - np.pi
        df["backtrack_severity"] = 0.0
        df["impairment_metric"] = 0.0
        df["grab_failed_event"] = 0.0

        intervals: list[dict[str, float]] = []
        traj_dist = 0.0
        traj_dur = 0.0

        for i in range(1, len(df)):
            dt = df.loc[i, "dt"]
            dx = df.loc[i, "dx"]
            dy = df.loc[i, "dy"]
            dist = np.sqrt(dx**2 + dy**2)
            angle_diff = np.abs(df.loc[i, "angle_diff"])

            if angle_diff > (3 * np.pi / 4):
                backtrack_dist = 0.0
                backtrack_dur = 0.0
                end_idx = i
                grab_fail_count = 0

                for j in range(i, len(df)):
                    end_idx = j
                    b_dt = df.loc[j, "dt"]
                    b_dx = df.loc[j, "dx"]
                    b_dy = df.loc[j, "dy"]
                    b_dist = np.sqrt(b_dx**2 + b_dy**2)

                    if backtrack_dur + b_dt > 1.0:
                        break

                    backtrack_dur += b_dt
                    backtrack_dist += b_dist

                    if j > i and np.abs(df.loc[j, "angle_diff"]) > (np.pi / 2):
                        grab_fail_count += 1

                if 0 < backtrack_dur < 1.0 and traj_dist > 0 and traj_dur > 0 and backtrack_dist < traj_dist:
                    severity = (backtrack_dist / traj_dist) * 100.0
                    impairment = min(severity / 100.0, 1.0)
                    grab_failed = 1.0 if grab_fail_count >= 2 else 0.0
                    intervals.append(
                        {
                            "t_start": df.loc[i, "t"],
                            "t_duration": backtrack_dur,
                            "backtrack_severity": severity,
                            "impairment_metric": impairment,
                            "grab_failed_event": grab_failed,
                        }
                    )
                    df.loc[i:end_idx, "backtrack_severity"] = severity
                    df.loc[i:end_idx, "impairment_metric"] = impairment
                    df.loc[i:end_idx, "grab_failed_event"] = grab_failed

                traj_dist = 0.0
                traj_dur = 0.0
            else:
                traj_dist += dist
                traj_dur += dt

        intervals_df = pd.DataFrame(intervals)
        if intervals_df.empty:
            intervals_df = pd.DataFrame(
                columns=["t_start", "t_duration", "backtrack_severity", "impairment_metric", "grab_failed_event"]
            )

        return df[["t", "backtrack_severity", "impairment_metric", "grab_failed_event"]], intervals_df


    @classmethod
    def master_df_to_intervals_df(cls, master_df: pd.DataFrame) -> pd.DataFrame:
        if master_df.empty:
            return pd.DataFrame(columns=["t_start", "t_duration", "session_index", "source_file_names", "parsed_filename_dt_start", "parsed_filename_dt_end"])
        grouped = master_df.groupby("session_index", sort=True)
        rows = []
        for session_index, grp in grouped:
            t_start_dt = pd.to_datetime(grp["parsed_filename_dt_start"].min())
            t_end_dt = pd.to_datetime(grp["parsed_filename_dt_end"].max())
            t_start_unix = float(datetime_to_unix_timestamp(t_start_dt))
            t_end_unix = float(datetime_to_unix_timestamp(t_end_dt))
            rows.append(
                {
                    "t_start": t_start_unix,
                    "t_duration": float(t_end_unix - t_start_unix),
                    "session_index": int(session_index),
                    "source_file_names": ", ".join(sorted(grp["source_file_name"].unique())),
                    "parsed_filename_dt_start": grp["parsed_filename_dt_start"].min(),
                    "parsed_filename_dt_end": grp["parsed_filename_dt_end"].max(),
                }
            )
        return pd.DataFrame(rows)


__all__ = ["IOGraphProcessor"]
