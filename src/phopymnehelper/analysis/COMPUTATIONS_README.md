# Computations

**Computations** are analysis steps that consume data from one or more sessions or streams and produce artifacts: time-resolved series, summaries, in-memory Python structures, and optionally dedicated views. The lists below describe what a computation may **access** as inputs and what it may **provide** as outputs. Orchestration (dependency graphs, caching, timeline integration) may live in notebooks or other packages; this document states the intended contract.

## Computations can access

1. **One or more datastreams or datasources** — For example, an EEG datasource attached to a session, or other modality streams needed for the analysis.

2. **A preceding required computation** — When a step depends on another step’s outputs, it declares that prerequisite (a dependency in a directed acyclic graph of computations).

3. **A computation cache** — A layer that stores prior results and reuses them when nothing material has changed, so rerunning a script or notebook on the same sessions does not repeat heavy work from scratch.

## Computations can provide

1. **A new datastream or datasource** — One or more values that are continuous or sampled in time, aligned with a timeline, plus metadata describing how they were produced (for example, a time-binned spectrogram from a subset of EEG channels over a chosen frequency range and analysis parameters).

2. **A new summary or aggregate statistic** — Session- or cohort-level scalars or small tables (for example, the average number of `HIGH_ACCEL` events in a session, or a session-averaged theta/delta power ratio restricted to frontal channels).

3. **Raw Python result objects** — Arbitrary structures (arrays, dicts, domain models) intended for downstream computations or for manual export to disk via your own serialization path.

4. **(Optional) A custom visualization or renderer** — A view that presents part of the output in context (for example, an EEG track that displays the spectrogram corresponding to a specific spectrogram computation).

## Implementations in this package

PhoPyMNEHelper currently hosts concrete analysis helpers rather than a full orchestration engine. Useful entry points:

- [`EEGComputations`](../EEG_data.py) — Batch-oriented helpers on `mne.io.Raw` (e.g. spectrogram, continuous wavelet transform, topo-style pipelines).
- [`analysis/computations/`](computations/) — Additional computation modules (for example, fatigue-related metrics in `fatigue_analysis.py`, and theta/delta sleep-intrusion style pipelines in `computations/specific/ADHD_sleep_intrusions.py`).

Caching, explicit dependency declarations between computations, and custom renderers are not defined solely in this folder; they are part of the broader contract above.
