# Changelog

## Unreleased

- Document the classic architecture, class map, and public API compatibility surface used by external front ends.
- Move the Gradio UI implementation under `ctor.ui` while preserving the `ctor.continuator_gradio` compatibility entry point.
- Move experimental helper modules under `ctor.legacy` while preserving the old `ctor.dynaprog` and `ctor.markov_analysis` imports.
- Add public API compatibility tests for the `Continuator2` generation surface and legacy imports.
- Move the constrained MIDI demo from `ctor.continuator` to `examples/constrained_note_continuator.py`.
- Fix realtime MIDI playback stop-state handling and mido `note_on` / `note_off` type checks.

## 1.2.3

- Remove unused legacy chord helper code from `mini_muse`.
- Compress the chord sequence example dataset and read it lazily from gzip.
- Remove stale chord-sequence demo code from `variable_order_markov`.
- Document the Hugging Face web client and local data policy.

## 1.2.2

- Merge the sparse chain-solver implementation into `main` as the constrained generation path.
- Remove the old generic belief-propagation graph implementation from the current core while keeping the public `NoSolutionErrorInBP` compatibility exception.
- Make identity viewpoint models skip realization address storage and compute zero-order priors from lightweight viewpoint counts.
- Keep realization storage enabled for computed viewpoints, including MIDI `Continuator2` models.
- Update solver tests to cover the sparse path directly.

## 1.2.1

- Fix Gradio launch compatibility.
