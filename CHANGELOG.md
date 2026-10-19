# Changelog

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
