# Changelog

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
