"""Tests for EventData Whisper annotation cleanup."""

import unittest

import pandas as pd

from phopymnehelper.event_data import EventData


class TestDropCommonErrorTranscriptions(unittest.TestCase):
    def test_drops_thanks_for_watching_and_ellipsis(self):
        df = pd.DataFrame({
            'onset': [1.0, 2.0, 3.0, 4.0],
            'duration': [0.0, 0.5, 0.0, 1.0],
            'description': [
                'real speech here',
                'Thanks for watching!',
                '...',
                'another real line',
            ],
        })
        out = EventData.drop_common_error_transcriptions(df)
        self.assertEqual(list(out['description']), ['real speech here', 'another real line'])
        self.assertEqual(len(out), 2)

    def test_drops_whitespace_and_case_variants(self):
        df = pd.DataFrame({
            'onset': [1.0, 2.0, 3.0],
            'duration': [0.0, 0.0, 0.0],
            'description': [
                'Thanks  for  watching!',
                'thanks for watching!',
                'keep me',
            ],
        })
        out = EventData.drop_common_error_transcriptions(df)
        self.assertEqual(list(out['description']), ['keep me'])

    def test_cleanup_whisper_calls_drop_after_dedup(self):
        # Consecutive duplicates so they survive the __len > 1 dedup filter,
        # then common-error drop should remove the hallucination phrase.
        df = pd.DataFrame({
            'onset': pd.to_datetime([
                '2026-01-01 00:00:00',
                '2026-01-01 00:00:01',
                '2026-01-01 00:00:02',
                '2026-01-01 00:00:03',
            ]),
            'duration': [0.0, 0.0, 0.0, 0.0],
            'description': [
                'keep this twice',
                'keep this twice',
                'Thanks for watching!',
                'Thanks for watching!',
            ],
            'WHISPER_idx': [0, 0, 0, 0],
            'filename': ['a.fif'] * 4,
            'file_meas_date': [pd.Timestamp('2026-01-01')] * 4,
        })
        out = EventData.perform_fixup_WHISPER_annotation_df(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]['description'], 'keep this twice')


if __name__ == '__main__':
    unittest.main()
