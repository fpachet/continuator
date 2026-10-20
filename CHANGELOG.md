# Changelog

## Unreleased

- Fix MIDI parsing mutation, concurrent track timing, channel/pitch pairing and
  overlapping repeated pitches. Share the parser across files and recordings.
- Honor export tempo, avoid accumulated tick truncation, and skip metadata on
  realtime MIDI output. Document beat units and incomplete-note handling.
- Remove retired relaxation keywords, decay/zero-order forwarding methods,
  unused constraint/Note helpers and the unused profiler package. Keep MIDI
  memory deletion in the facade and format traces from the symbolic owner.
- Pin an upstream cold positional-inference optimization, preserving numerical
  underflow support. Add explicit post-learning timing to the audit.
- Add Python 3.11/3.13 CI, wheel/sdist builds and isolated installed-wheel tests.

- Remove compatibility-only `ctor.continuator_gradio`, `ctor.midi_ports_poll`,
  `ctor.vo_regular_bp_continuator` and `ctor.variable_order_markov` modules.
  Use `ctor.ui.gradio_app`, `ctor.ui.midi_ports_poll`, `ctor.vo_regular_bp`
  and `ctor.sequence_model` directly; migration examples are documented.
- Move the local MIDI listener to `ctor.ui.phrase_listener`. Keep the stable
  Continuator2 import and public infeasibility exception.
- Update the sibling frontend contract test for its single-engine API.

## 2.0.0

- Maintain only the VO-Regular-BP engine. Archive classic, context-BP, their
  solver/selection wrappers and dedicated tests in the recovery tag
  `codex/archive-three-engines-2026-10-19`.
- Keep `ctor.continuator.Continuator2` stable and introduce `SequenceModel` for
  generic symbols. Keep a deprecated generation-only Variable_order_Markov
  adapter; retire classic counter, decay and marginal APIs explicitly.
- Fix free-initial generation, emitted START constraints and final note/END unions.
- Apply viewpoint functions consistently during repeated generic learning, even
  when original objects equal existing viewpoints.
- Reuse prepared inference with bounded LRU admission and invalidate it on
  memory/configuration changes. Defer model rebuilding until first use.
- Pin the backend revision containing shared first-hit inference. Deduplicate
  equivalent MIDI realization domains for exact duplicate phrases.
- Migrate examples, bound the default chord demo corpus, remove the inactive
  Gradio decay control and synchronize package versions.
- Add generation, invalidation, memory, serialization and real frontend contract
  tests, plus reproducible before/after performance measurements.

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
