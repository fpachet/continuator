# Current Class Map

This file is a concise class-level index for the current implementation.

## Generic Model and Inference

### `ctor.variable_order_markov.Variable_order_Markov`

Generic classic variable-order Markov model. It learns sequences of arbitrary
objects or computed viewpoints, stores context continuation counts up to
`kmax`, builds a first-order transition matrix, and exposes constrained
generation methods.

Use it directly for non-MIDI examples such as integers, characters, words, and
chord symbols.

Key public methods:

- `learn_sequence(sequence_of_stuff)`
- `sample_sequence(length, prefix=None, constraints=None, start_vp=None, ...)`
- `continue_sequence(prefix, length, constraints=None, ...)`
- `continue_until_end(prefix=None, min_length=1, max_length=64, end_vp=None)`
- `chain_marginals(length, constraints=None)`
- `get_first_order_matrix()`
- `set_period_mode(mode)`

### `ctor.variable_order_markov.LazyExpCounter`

Per-key lazy exponential counter. It supports decayed weights without eagerly
updating every key at each time step.

### `ctor.variable_order_markov.MultiCounter`

Continuation-count container for one context. It combines exact counts with
fast and slow decayed counts and exposes `full`, `late`, `middle`, and `early`
views.

### `ctor.chain_solver.SparseForwardBackward`

Iterative sparse forward-backward solver for a first-order chain with transition
matrix convention `P[previous, next]`.

Key public methods:

- `forward_backward(unary_potentials)`
- `reachable_to_target(target_index, max_steps)`
- `first_hit_reachable_to_target(target_index, max_steps)`
- `can_reach_between(reachable, state_index, min_steps, max_steps)`

### `ctor.chain_solver.ForwardBackwardResult`

Dataclass containing:

- `marginals`
- `forward`
- `backward`

### `ctor.chain_solver.NoSolutionErrorInChainSolver`

Raised when transition weights and unary constraints leave no feasible chain.

### `ctor.belief_propag.NoSolutionErrorInBP`

Compatibility exception used by public APIs. Despite the name, the old generic
belief-propagation graph is no longer present in the current core.

## Constraints

### `ctor.constraints.ConstraintProblem`

Small builder for positional constraints over generated sequences.

Example:

```python
problem = ConstraintProblem(length=16)
problem.at(0).equals("C")
problem.at(8).one_of(["F", "Dm"])
problem.at(15).equals("C")
```

### `ctor.constraints.PositionConstraint`

Helper returned by `ConstraintProblem.at(position)`. It provides `equals` and
`one_of`.

## MIDI Application Layer

### `ctor.continuator.Continuator2`

High-level MIDI Continuator facade. It wraps a `Variable_order_Markov` configured
with a MIDI-note viewpoint extractor.

Key public methods:

- `learn_file(midi_file, transposition)`
- `learn_folder(folder_path, transpose=False)`
- `learn_phrase(note_sequence, transposition)`
- `learn_phrase_from_mido(phrase)`
- `sample_sequence(...)`
- `continue_sequence(...)`
- `continue_until_end(...)`
- `realize_vp_sequence(vp_seq)`
- `save_midi(sequence, output_file, tempo=120, sustain=False)`

### `midi_stuff.mini_muse.Note`

Lightweight note object with pitch, velocity, duration, start time, and neighbor
delta information. `Continuator2` uses its overlap methods to compute
viewpoints and reconstruct timing.

### `ctor.phrase_listener.MidiPhraseListener`

Realtime MIDI helper that records incoming MIDI messages into phrases and can
play phrases back through an output port.

### `ctor.ui.gradio_app.Continuator_gradio`

Local Gradio UI wrapper around `Continuator2`, MIDI ports, phrase memory, piano
roll display, and generation controls.

The older `ctor.continuator_gradio.Continuator_gradio` import remains as a
compatibility wrapper.

## Utilities and Experimental Helpers

### `ctor.legacy.dynaprog.VariableDomainSequenceOptimizer`

Generic dynamic-programming optimizer over a sequence of variable domains. It is
currently available for realization experiments but is not the default MIDI
realizer.

The older `ctor.dynaprog.VariableDomainSequenceOptimizer` import remains as a
compatibility wrapper.

### `ctor.legacy.markov_analysis.MarkovAnalysis`

Dataclass returned by `analyze_markov_chain`. It stores Markov-chain diagnostics
such as irreducibility, period, ergodicity, stationary distribution, strongly
connected components, and optional primitive exponent.

The older `ctor.markov_analysis` import remains as a compatibility wrapper.

## Practical Import Guide

For the current generic model:

```python
from ctor.variable_order_markov import Variable_order_Markov
from ctor.constraints import ConstraintProblem
```

For the current MIDI Continuator:

```python
from ctor.continuator import Continuator2
```

For direct sparse chain inference:

```python
from ctor.chain_solver import SparseForwardBackward
```

For future redesign work, treat this current set of classes as the "classic"
implementation to preserve and compare against.

For external compatibility details, see `docs/public_api.md`.
