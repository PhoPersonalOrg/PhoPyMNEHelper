"""Session- or cohort-specific analysis helpers (fatigue, ADHD/sleep intrusions, etc.)."""

from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import ThetaDeltaSleepIntrusionComputation
from phopymnehelper.analysis.computations.specific.bad_epochs import BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME, BAD_EPOCHS_QC_PARAM_KEYS, BadEpochsQCComputation
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase
from phopymnehelper.analysis.computations.specific.EEG_Spectograms import DEFAULT_SPECTROGRAM_NOVERLAP, DEFAULT_SPECTROGRAM_NPERSEG, EEG_SPECTROGRAM_PARAM_KEYS, EEGSpectrogramComputation
from phopymnehelper.analysis.computations.specific.frontal_midline_theta import FrontalMidlineThetaComputation
from phopymnehelper.analysis.computations.specific.jaw_clench_probability import JawClenchProbabilityComputation
from phopymnehelper.analysis.computations.specific.mental_states import FrameMentalStatesComputation

__all__ = [
    "BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME",
    "BAD_EPOCHS_QC_PARAM_KEYS",
    "BadEpochsQCComputation",
    "DEFAULT_SPECTROGRAM_NOVERLAP",
    "DEFAULT_SPECTROGRAM_NPERSEG",
    "EEG_SPECTROGRAM_PARAM_KEYS",
    "EEGSpectrogramComputation",
    "FrameMentalStatesComputation",
    "FrontalMidlineThetaComputation",
    "JawClenchProbabilityComputation",
    "SpecificComputationBase",
    "ThetaDeltaSleepIntrusionComputation",
]
