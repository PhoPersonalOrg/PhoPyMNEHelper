"""Tests for phopymnehelper.iograph_data.IOGraphProcessor."""

import os
import tempfile
from pathlib import Path

import pandas as pd

from phopymnehelper.iograph_data import IOGraphProcessor


def _write_csv(path: Path, times_ms: list[float], xs: list[float], ys: list[float], mtime: pd.Timestamp) -> None:
    df = pd.DataFrame({"time": times_ms, "x": xs, "y": ys})
    df.to_csv(path, index=False)
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))


def test_parsed_filename_dt_range_simple() -> None:
    name = "IOGraphica - 1 hour (from 14-37 to 15-40).csv"
    modified = pd.Timestamp("2026-05-12 19:40:17")
    dt_start, dt_end = IOGraphProcessor._parsed_filename_dt_range(name, modified)
    assert dt_start == pd.Timestamp("2026-05-12 14:37:00")
    assert dt_end == pd.Timestamp("2026-05-12 15:40:00")


def test_parsed_filename_dt_range_dated() -> None:
    name = "IOGraphica - 1.3 hours (from 22-56 May 11th to 0-16 May 12th).csv"
    modified = pd.Timestamp("2026-05-12 04:16:17")
    dt_start, dt_end = IOGraphProcessor._parsed_filename_dt_range(name, modified)
    assert dt_start == pd.Timestamp("2026-05-11 22:56:00")
    assert dt_end == pd.Timestamp("2026-05-12 00:16:00")


def test_overlapping_files_merge_and_dedupe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mtime = pd.Timestamp("2026-01-01 12:00:00")
        long_path = root / "IOGraphica - 1 hour (from 10-00 to 11-00).csv"
        short_path = root / "IOGraphica - 30 minutes (from 10-30 to 11-00).csv"
        _write_csv(long_path, [0, 1_800_000, 3_600_000], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], mtime)
        _write_csv(short_path, [0, 1_800_000], [10.0, 20.0], [40.0, 50.0], mtime)
        file_table, _ = IOGraphProcessor.scan_csv_directory(root, recursive=False)
        master_df, _ = IOGraphProcessor.build_master_df(file_table, drop_na_coords=True)
        assert master_df["session_index"].nunique() == 1
        assert master_df.duplicated(subset=["session_index", "sample_time"]).sum() == 0
        overlap_rows = master_df[master_df["x"] == 2.0]
        assert len(overlap_rows) == 1
        assert overlap_rows.iloc[0]["source_file_name"] == long_path.name


def test_incremental_saves_same_start_shorter_wins() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mtime = pd.Timestamp("2026-01-01 12:00:00")
        short_path = root / "IOGraphica - 30 minutes (from 10-00 to 10-30).csv"
        long_path = root / "IOGraphica - 1 hour (from 10-00 to 11-00).csv"
        _write_csv(short_path, [0, 900_000, 1_800_000], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], mtime)
        _write_csv(long_path, [0, 900_000, 1_800_000, 3_600_000], [10.0, 20.0, 30.0, 40.0], [40.0, 50.0, 60.0, 70.0], mtime)
        file_table, _ = IOGraphProcessor.scan_csv_directory(root, recursive=False)
        master_df, _ = IOGraphProcessor.build_master_df(file_table, drop_na_coords=True)
        assert master_df["session_index"].nunique() == 1
        assert master_df.duplicated(subset=["session_index", "sample_time"]).sum() == 0
        overlap_rows = master_df[master_df["sample_time"] == master_df["sample_time"].sort_values().iloc[1]]
        assert len(overlap_rows) == 1
        assert overlap_rows.iloc[0]["source_file_name"] == short_path.name
        assert overlap_rows.iloc[0]["x"] == 2.0
        tail_rows = master_df[master_df["x"] == 40.0]
        assert len(tail_rows) == 1
        assert tail_rows.iloc[0]["source_file_name"] == long_path.name


def test_master_df_to_detail_and_intervals() -> None:
    master_df = pd.DataFrame(
        {
            "session_index": [0, 0],
            "sample_time": [pd.Timestamp("2026-01-01 10:00:00"), pd.Timestamp("2026-01-01 10:00:01")],
            "time": [0, 1000],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "source_file_name": ["a.csv", "a.csv"],
            "parsed_filename_dt_start": [pd.Timestamp("2026-01-01 10:00:00"), pd.Timestamp("2026-01-01 10:00:00")],
            "parsed_filename_dt_end": [pd.Timestamp("2026-01-01 10:00:01"), pd.Timestamp("2026-01-01 10:00:01")],
        }
    )
    detail_df = IOGraphProcessor.master_df_to_detail_df(master_df)
    assert list(detail_df.columns) == ["t", "x", "y", "session_index", "source_file_name"]
    assert len(detail_df) == 2
    intervals_df = IOGraphProcessor.master_df_to_intervals_df(master_df)
    assert list(intervals_df.columns) == ["t_start", "t_duration", "session_index", "source_file_names", "parsed_filename_dt_start", "parsed_filename_dt_end"]
    assert len(intervals_df) == 1
