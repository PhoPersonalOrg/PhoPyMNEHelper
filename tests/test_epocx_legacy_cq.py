"""Unit tests for legacy CyKit nibble CQ reconstruction, broadcast guard, MNE channel types, and step-hold gating."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from phopylslhelper.core.data_modalities import MOTION_CHANNELS_EPOCX, MOTION_CHANNELS_LEGACY_GYRO, lab_recorder_to_mne_to_type_dict, modality_channels_dict, modality_sfreq_dict
from phopymnehelper.analysis.computations.eeg_manifold import apply_quality_and_session_gates
from phopymnehelper.epocx_legacy_cq import (
    CQ_CHANNEL_NAMES,
    CQ_SLOT_SCALE_DIVISORS,
    CQ_SLOT_TO_1020_IDX,
    cq_norm_to_score,
    cq_score_to_norm,
    is_corrupted_broadcast_equality,
    is_legacy_nibble_equality,
    reconstruct_legacy_equality_array,
)
from phopymnehelper.xdf_files import LabRecorderXDF


def _pack_nibbles_to_word7(raw_u16: int) -> Tuple[int, int, int, int]:
    """Encode a 16-bit raw `words[7]` integer into 4 legacy CyKit `plain[16..17]` nibbles."""
    w7 = int(raw_u16) & 0xFFFF
    return (w7 & 0xF, (w7 >> 4) & 0xF, (w7 >> 8) & 0xF, (w7 >> 12) & 0xF)


def test_cq_norm_score_roundtrip_and_breakpoints() -> None:
    assert cq_norm_to_score(0.0) == pytest.approx(0.0)
    assert cq_norm_to_score(0.2) == pytest.approx(1.0)
    assert cq_norm_to_score(0.7) == pytest.approx(2.0)
    assert cq_norm_to_score(1.7) == pytest.approx(3.0)
    assert cq_norm_to_score(2.7) == pytest.approx(4.0)

    for norm in (0.0, 0.1, 0.2, 0.45, 0.7, 1.2, 1.7, 3.5, 8.2):
        score = cq_norm_to_score(norm)
        assert cq_score_to_norm(score) == pytest.approx(norm, abs=1e-6)
    ## END for norm in (0.0, 0.1, 0.2, 0.45, 0.7, 1.2, 1.7, 3.5, 8.2)...


def test_reconstruct_legacy_cykit_nibble_equality_recovers_per_electrode_scores() -> None:
    # Distinct raw words[7] impedance levels across the 14 slots (0..13):
    slot_word7 = [0, 350, 900, 1800, 1950, 2100, 2250, 2400, 2550, 2700, 2850, 3000, 3200, 3600]
    n_samples = 512
    start_counter = 37
    # Real legacy CyKit XDF had 14 nibble columns: cols 0..3 = words[7], cols 4..13 = EEG O2/P8/T8 nibbles (MSB ~8)
    eq_14ch = np.zeros((n_samples, 14), dtype=np.float64)
    eq_14ch[:, 7] = 8.0
    eq_14ch[:, 11] = 8.0
    for i in range(n_samples):
        c = (start_counter + i) % 128
        if c < 14:
            raw_val = slot_word7[c]
        elif 16 <= c < 64:
            raw_val = 0  # idle zero window
        elif 64 <= c < 78:
            raw_val = slot_word7[c - 64]
        elif 82 <= c < 127:
            raw_val = (1600, 1620, 1640, 1660)[(c - 82) % 4]
        elif c == 127:
            raw_val = 84  # battery %
        else:
            raw_val = 0
        eq_14ch[i, :4] = _pack_nibbles_to_word7(raw_val)
    ## END for i in range(n_samples)...

    assert is_legacy_nibble_equality(eq_14ch)
    assert not is_corrupted_broadcast_equality(eq_14ch)

    rec = reconstruct_legacy_equality_array(eq_14ch, sfreq=128.0)
    assert rec.was_legacy_nibble is True
    assert rec.was_corrupted_broadcast is False
    assert rec.matched_fraction > 0.95
    assert rec.cq_scores.shape == (n_samples, 14)
    assert np.nanmax(rec.battery_percent) == pytest.approx(84.0)

    # After the first 128-packet cycle (1.0 s), every 10-20 electrode holds its exact calibrated continuous score
    tail = rec.cq_scores[160:, :]
    for slot_idx, raw_val in enumerate(slot_word7):
        ch_1020_idx = CQ_SLOT_TO_1020_IDX[slot_idx]
        expected_norm = (float(raw_val) * 0.25) / CQ_SLOT_SCALE_DIVISORS[slot_idx]
        expected_score = cq_norm_to_score(expected_norm)
        np.testing.assert_allclose(tail[:, ch_1020_idx], expected_score, atol=1e-6)
    ## END for slot_idx, raw_val in enumerate(slot_word7)...


def test_corrupted_broadcast_equality_guard_returns_nan() -> None:
    # Simulate Nov 11-26 `float(words[7] & 0x7F)` broadcast across all 14 channels (with >=35% idle zeros)
    col = np.zeros((256, 1), dtype=np.float64)
    for i in range(256):
        c = i % 128
        if c < 14 or (64 <= c < 78):
            col[i, 0] = float((500 + 13 * c) & 0x7F)
        elif c == 127:
            col[i, 0] = 78.0
    ## END for i in range(256)...

    broadcast_14ch = np.repeat(col, 14, axis=1)
    assert is_corrupted_broadcast_equality(broadcast_14ch)
    rec = reconstruct_legacy_equality_array(broadcast_14ch)
    assert rec.was_corrupted_broadcast is True
    assert rec.was_legacy_nibble is False
    assert np.all(np.isnan(rec.cq_scores))


def test_modality_defaults_and_xml_channel_label_extraction() -> None:
    assert modality_channels_dict['MOTION'] == MOTION_CHANNELS_EPOCX
    assert len(MOTION_CHANNELS_EPOCX) == 10
    assert len(MOTION_CHANNELS_LEGACY_GYRO) == 6
    assert modality_sfreq_dict['MOTION'] == 32
    assert lab_recorder_to_mne_to_type_dict['QUAT'] == 'misc'
    assert lab_recorder_to_mne_to_type_dict['MAG'] == 'misc'
    assert lab_recorder_to_mne_to_type_dict['CQ'] == 'misc'

    eq_info = {
        'desc': [{
            'channels': [{
                'channel': [{'label': [f'q{ch}'], 'type': ['CQ']} for ch in CQ_CHANNEL_NAMES],
            }],
        }],
    }
    extracted_eq = LabRecorderXDF.extract_channel_labels_from_stream_info(eq_info, 'CQ', 'Epoc X eQuality', 14)
    assert extracted_eq == list(CQ_CHANNEL_NAMES)

    motion_10_info = {
        'desc': [{
            'channels': [{
                'channel': [{'label': [ch], 'type': ['QUAT']} for ch in MOTION_CHANNELS_EPOCX],
            }],
        }],
    }
    assert LabRecorderXDF.extract_channel_labels_from_stream_info(motion_10_info, 'Signal', 'Epoc X Motion', 10) == MOTION_CHANNELS_EPOCX
    assert LabRecorderXDF.extract_channel_labels_from_stream_info({}, 'Signal', 'Epoc X Motion', 6) == MOTION_CHANNELS_LEGACY_GYRO


def test_manifold_quality_gate_uses_step_hold_not_linear_ramp() -> None:
    from phopymnehelper.analysis.computations.eeg_manifold import EEGManifoldConfig

    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float64)
    eeg_df = pd.DataFrame({'t': times, 'AF3': np.ones(len(times), dtype=np.float64)})
    # Quality jumps from 0.0 at t=0.0 to 4.0 at t=1.0: under step-hold, t=0.5 and t=0.75 stay 0.0 (< 2.0 -> NaN)!
    q_df = pd.DataFrame({'t': [0.0, 1.0], 'AF3': [0.0, 4.0]})
    cfg = EEGManifoldConfig(min_quality_to_keep=2.0, mask_unknown_quality=True)
    gated_df, usable = apply_quality_and_session_gates(eeg_df, q_df, ['AF3'], cfg)
    assert usable == ['AF3']
    assert np.all(np.isnan(gated_df['AF3'].to_numpy()[:4]))
    assert np.all(np.isfinite(gated_df['AF3'].to_numpy()[4:]))
