"""Tests for phopymnehelper.iograph_data.IOGraphProcessor."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from phopymnehelper.iograph_data import IOGraphProcessor


def _write_csv(path: Path, times_ms: list[float], xs: list[float], ys: list[float], mtime: pd.Timestamp) -> None:
    df = pd.DataFrame({"time": times_ms, "x": xs, "y": ys})
    df.to_csv(path, index=False)
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))


class TestIOGraphParsing(unittest.TestCase):
    def test_parsed_filename_dt_range_simple(self) -> None:
        name = "IOGraphica - 1 hour (from 14-37 to 15-40).csv"
        modified = pd.Timestamp("2026-05-12 19:40:17")
        dt_start, dt_end = IOGraphProcessor._parsed_filename_dt_range(name, modified)
        self.assertEqual(dt_start, pd.Timestamp("2026-05-12 14:37:00"))
        self.assertEqual(dt_end, pd.Timestamp("2026-05-12 15:40:00"))

    def test_parsed_filename_dt_range_dated(self) -> None:
        name = "IOGraphica - 1.3 hours (from 22-56 May 11th to 0-16 May 12th).csv"
        modified = pd.Timestamp("2026-05-12 04:16:17")
        dt_start, dt_end = IOGraphProcessor._parsed_filename_dt_range(name, modified)
        self.assertEqual(dt_start, pd.Timestamp("2026-05-11 22:56:00"))
        self.assertEqual(dt_end, pd.Timestamp("2026-05-12 00:16:00"))

    def test_parsed_filename_dt_range_absolute(self) -> None:
        name = "IOGraphica - 2026-11-09_12-34-12_14.8083h.csv"
        dt_start, dt_end = IOGraphProcessor._parsed_filename_dt_range(name, pd.NaT)
        self.assertEqual(dt_start, pd.Timestamp("2026-11-09 12:34:12"))
        self.assertEqual(dt_end, dt_start + pd.Timedelta(hours=14.8083))

    def test_parsed_filename_dt_range_absolute_sample_basename(self) -> None:
        sample = Path(__file__).resolve().parents[2] / "IOGraph" / "data" / "test_output" / "IOGraphica - 2026-11-09_12-34-12_14.8083h.csv"
        name = sample.name if sample.exists() else "IOGraphica - 2026-11-09_12-34-12_14.8083h.csv"
        dt_start, dt_end = IOGraphProcessor._parsed_filename_dt_range(name, pd.NaT)
        self.assertEqual(dt_start, pd.Timestamp("2026-11-09 12:34:12"))
        self.assertEqual(dt_end, dt_start + pd.Timedelta(hours=14.8083))


class TestIOGraphMerge(unittest.TestCase):
    def test_overlapping_files_merge_and_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mtime = pd.Timestamp("2026-01-01 12:00:00")
            long_path = root / "IOGraphica - 1 hour (from 10-00 to 11-00).csv"
            short_path = root / "IOGraphica - 30 minutes (from 10-30 to 11-00).csv"
            _write_csv(long_path, [0, 1_800_000, 3_600_000], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], mtime)
            _write_csv(short_path, [0, 1_800_000], [10.0, 20.0], [40.0, 50.0], mtime)
            file_table, _ = IOGraphProcessor.scan_csv_directory(root, recursive=False)
            master_df, _ = IOGraphProcessor.build_master_df(file_table, drop_na_coords=True)
            self.assertEqual(master_df["session_index"].nunique(), 1)
            self.assertEqual(master_df.duplicated(subset=["session_index", "sample_time"]).sum(), 0)
            overlap_rows = master_df[master_df["x"] == 2.0]
            self.assertEqual(len(overlap_rows), 1)
            self.assertEqual(overlap_rows.iloc[0]["source_file_name"], long_path.name)

    def test_incremental_saves_same_start_shorter_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mtime = pd.Timestamp("2026-01-01 12:00:00")
            short_path = root / "IOGraphica - 30 minutes (from 10-00 to 10-30).csv"
            long_path = root / "IOGraphica - 1 hour (from 10-00 to 11-00).csv"
            _write_csv(short_path, [0, 900_000, 1_800_000], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], mtime)
            _write_csv(long_path, [0, 900_000, 1_800_000, 3_600_000], [10.0, 20.0, 30.0, 40.0], [40.0, 50.0, 60.0, 70.0], mtime)
            file_table, _ = IOGraphProcessor.scan_csv_directory(root, recursive=False)
            master_df, _ = IOGraphProcessor.build_master_df(file_table, drop_na_coords=True)
            self.assertEqual(master_df["session_index"].nunique(), 1)
            self.assertEqual(master_df.duplicated(subset=["session_index", "sample_time"]).sum(), 0)
            overlap_rows = master_df[master_df["sample_time"] == master_df["sample_time"].sort_values().iloc[1]]
            self.assertEqual(len(overlap_rows), 1)
            self.assertEqual(overlap_rows.iloc[0]["source_file_name"], short_path.name)
            self.assertEqual(overlap_rows.iloc[0]["x"], 2.0)
            tail_rows = master_df[master_df["x"] == 40.0]
            self.assertEqual(len(tail_rows), 1)
            self.assertEqual(tail_rows.iloc[0]["source_file_name"], long_path.name)

    def test_absolute_incremental_saves_same_start_keep_longest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mtime = pd.Timestamp("2026-11-09 20:00:00")
            short_path = root / "IOGraphica - 2026-11-09_10-00-00_0.5000h.csv"
            long_path = root / "IOGraphica - 2026-11-09_10-00-00_1.0000h.csv"
            _write_csv(short_path, [0, 900_000, 1_800_000], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], mtime)
            _write_csv(long_path, [0, 900_000, 1_800_000, 3_600_000], [10.0, 20.0, 30.0, 40.0], [40.0, 50.0, 60.0, 70.0], mtime)
            file_table, _ = IOGraphProcessor.scan_csv_directory(root, recursive=False)
            self.assertTrue(file_table["parsed_filename_dt_start"].notna().all())
            master_df, _ = IOGraphProcessor.build_master_df(file_table, drop_na_coords=True)
            self.assertEqual(master_df["session_index"].nunique(), 1)
            self.assertEqual(len(master_df), 4)
            self.assertEqual(master_df.duplicated(subset=["session_index", "sample_time"]).sum(), 0)
            self.assertTrue((master_df["source_file_name"] == long_path.name).all())
            self.assertEqual(master_df.iloc[0]["sample_time"], pd.Timestamp("2026-11-09 10:00:00"))
            self.assertEqual(master_df.iloc[0]["x"], 10.0)
            overlap_time = pd.Timestamp("2026-11-09 10:15:00")
            overlap_rows = master_df[master_df["sample_time"] == overlap_time]
            self.assertEqual(len(overlap_rows), 1)
            self.assertEqual(overlap_rows.iloc[0]["source_file_name"], long_path.name)
            self.assertEqual(overlap_rows.iloc[0]["x"], 20.0)
            tail_time = pd.Timestamp("2026-11-09 11:00:00")
            tail_rows = master_df[master_df["sample_time"] == tail_time]
            self.assertEqual(len(tail_rows), 1)
            self.assertEqual(tail_rows.iloc[0]["source_file_name"], long_path.name)
            self.assertEqual(tail_rows.iloc[0]["x"], 40.0)


    def test_absolute_real_fixtures_same_start_keep_longest(self) -> None:
        fixture_dir = Path(__file__).resolve().parents[2] / "IOGraph" / "data" / "test_output"
        short_src = fixture_dir / "IOGraphica - 2026-11-09_12-34-12_14.8083h.csv"
        long_src = fixture_dir / "IOGraphica - 2026-11-09_12-34-12_15.7639h.csv"
        if not short_src.exists() or not long_src.exists():
            self.skipTest(f"Missing real fixtures under {fixture_dir}")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            short_dst = root / short_src.name
            long_dst = root / long_src.name
            short_dst.write_bytes(short_src.read_bytes())
            long_dst.write_bytes(long_src.read_bytes())
            long_valid = pd.read_csv(long_dst).dropna(subset=["x", "y"])
            expected_rows = len(long_valid)
            expected_first_time = float(long_valid["time"].iloc[0])
            file_table, _ = IOGraphProcessor.scan_csv_directory(root, recursive=False)
            master_df, _ = IOGraphProcessor.build_master_df(file_table, drop_na_coords=True)
            self.assertEqual(master_df["session_index"].nunique(), 1)
            self.assertEqual(master_df.duplicated(subset=["session_index", "sample_time"]).sum(), 0)
            self.assertEqual(len(master_df), expected_rows)
            self.assertTrue((master_df["source_file_name"] == long_src.name).all())
            dt_start = pd.Timestamp("2026-11-09 12:34:12")
            first_sample = master_df["sample_time"].min()
            expected_first_sample = dt_start + pd.Timedelta(milliseconds=expected_first_time)
            self.assertLessEqual(abs((first_sample - expected_first_sample).total_seconds()), 0.001)


class TestIOGraphAdapters(unittest.TestCase):
    def test_master_df_to_detail_and_intervals(self) -> None:
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
        self.assertEqual(list(detail_df.columns), ["t", "x", "y", "session_index", "source_file_name"])
        self.assertEqual(len(detail_df), 2)
        intervals_df = IOGraphProcessor.master_df_to_intervals_df(master_df)
        self.assertEqual(
            list(intervals_df.columns),
            ["t_start", "t_duration", "session_index", "source_file_names", "parsed_filename_dt_start", "parsed_filename_dt_end"],
        )
        self.assertEqual(len(intervals_df), 1)


if __name__ == "__main__":
    unittest.main()
