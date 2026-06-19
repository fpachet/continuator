# Current Class Map

This file is a concise class-level index for the current implementation.

## Generic Model and Inference

### `ctor.context_bp.ContextBPModel`

Experimental generic variable-order model with exact context-state
forward-backward inference. It learns arbitrary symbolic sequences, compiles
variable-order continuation counts into a sparse context graph, and samples
fixed-length sequences with positional hard constraints over emitted symbols.
Constrained sampling uses stepwise order backoff and an explicit order policy.
By default it uses `LongestFeasiblePolicy`, which tries the longest feasible
context at each emitted position.

This is the first implementation of the new context-BP engine and currently
lives beside the classic implementation.

Key public methods:

- `learn_sequence(sequence)`
- `infer(length, prefix=None, constraints=None)`
- `symbol_marginals(length, prefix=None, constraints=None)`
- `sample_sequence(length, prefix=None, constraints=None, initial_mode="start", raise_on_fail=False)`
- `sample_sequence_with_trace(length, prefix=None, constraints=None, initial_mode="start", raise_on_fail=False)`
- `continue_until_end(prefix=None, min_length=1, max_length=64, end_symbol=None)`
- `continue_until_end_with_trace(prefix=None, min_length=1, max_length=64, end_symbol=None)`
- `first_hit_lengths(prefix=None, min_length=1, max_length=64, end_symbol=None)`
- `last_sample_trace_as_dicts()`

`initial_mode="start"` uses the hidden START context. `initial_mode="free"`
chooses the first emitted symbol from learned symbols that can still satisfy
the remaining constraints, then continues with normal stepwise order policy.

### `ctor.context_bp.SampleStep`

Dataclass used by `sample_sequence_with_trace`. It records the generated
position, symbol, chosen suffix order, selected graph order, and context used
for that step. It also records the order policy name, candidate orders,
candidate counts, skipped singleton orders, skipped symbol, and whether the
selected singleton was explicitly accepted.

### `ctor.context_bp.LongestFeasiblePolicy`

Default generic context-BP order policy. It chooses the first feasible
candidate set produced by the stepwise backoff search, i.e. the longest context
compatible with the remaining constraints.

### `ctor.context_bp.SingletonAvoidingBackoffPolicy`

Classic Continuator-style practical policy. It usually skips higher-order
contexts that have only one feasible continuation after BP/constraint
filtering, suppresses that skipped symbol at intermediate lower orders when
alternatives exist, and leaves order 1 as the final fallback.

Constructor parameters:

- `acceptance_probability`: optional fixed probability in `[0, 1]`; `None`
  uses the classic `1 / (order + 1)` rule.
- `min_singleton_order`: lowest context order eligible for singleton skipping.
- `suppress_skipped_symbol`: whether to mask a skipped symbol at intermediate
  lower orders when alternatives exist.

### `ctor.context_bp.ContextBPResult`

Dataclass containing context-BP messages, emitted-symbol marginals, and the
total path mass for the constrained finite chain.

### `ctor.engines.ClassicSequenceEngine`

Thin generic adapter around `Variable_order_Markov`. It gives the classic
engine the same basic method names as the experimental context-BP engine.

### `ctor.engines.ContextBPSequenceEngine`

Thin generic adapter around `ContextBPModel`.

### `ctor.engines.make_sequence_engine`

Factory for choosing the generic sequence engine explicitly:

```python
from ctor.engines import make_sequence_engine

engine = make_sequence_engine("classic")
engine = make_sequence_engine("context_bp")
```

### `ctor.classic.variable_order_markov.Variable_order_Markov`

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

### `ctor.classic.variable_order_markov.LazyExpCounter`

Per-key lazy exponential counter. It supports decayed weights without eagerly
updating every key at each time step.

### `ctor.classic.variable_order_markov.MultiCounter`

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

### `ctor.classic.continuator.ClassicContinuator`

High-level classic MIDI Continuator facade. It wraps a
`Variable_order_Markov` configured with a MIDI-note viewpoint extractor and uses
shared MIDI utilities from `ctor.midi`.

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

### `ctor.classic.continuator.Continuator2`

Historical compatibility subclass of `ClassicContinuator` inside the classic
package. The top-level default `ctor.continuator.Continuator2` now uses the
VO-Regular-BP backend.

### `ctor.context_bp.ContextBPContinuator`

Experimental MIDI Continuator facade. It delegates viewpoint generation to
`ContextBPModel` and keeps only a lightweight `MidiRealizationStore` for MIDI
realization. It does not subclass `Continuator2`.

Use this class for context-BP MIDI experiments while keeping it separate from
the default `Continuator2` entry point. It uses
`SingletonAvoidingBackoffPolicy` by default to preserve the classic
Continuator's practical anti-copying behavior.
It exposes `sample_sequence_with_trace`, `continue_until_end_with_trace`, and
`get_last_generation_trace` for debugging the selected orders and singleton
backoffs. When called without a prefix or explicit handoff viewpoint, its
fixed-length generation uses context-BP's free-initial mode, matching the
classic engine's memory-only generation more closely.

### `ctor.vo_regular_bp.VORegularBPContinuator`

Experimental MIDI Continuator facade backed by the external `vo_regular_bp`
library. It uses an order-stack model for viewpoint generation and a separate
MIDI realization store. It supports fixed-length positional constraints,
generalized first-hit `continue_until(...)`, `continue_until_end(...)`, and
explicit or virtual transposition augmentation. Virtual transposition keeps the
symbolic model virtual and realizes transformed viewpoints lazily by applying
the corresponding note transform to learned base notes.

Use this class directly for regular-constrained generation and
data-augmentation semantics. The top-level default
`ctor.continuator.Continuator2` subclasses this facade.

### `ctor.midi.MidiContinuatorBase`

Shared MIDI helper base for MIDI-facing facades. It contains MIDI parsing,
MIDI writing, phrase conversion, default note viewpoint extraction,
transposition helpers, and viewpoint-to-note realization. The default realizer
uses a vectorized dynamic-programming pass over viewpoint-specific address
domains, so all MIDI engines share the same policy for choosing concrete
learned notes. It prefers compatible overlap state, consecutive source
addresses, and stable transform choices; generated END markers force the last
realized note to come from an ending address when one is available.

### `ctor.midi.MidiRealizationStore`

Small realization memory for MIDI facades. It stores learned note phrases and
maps each viewpoint to source note addresses, without variable-order sampling,
transition counts, or decay modes.

### `midi_stuff.mini_muse.Note`

Lightweight note object with pitch, velocity, duration, start time, a
`next_start_delta` timing offset, and an `overlaps_left` flag. `Continuator2`
uses its overlap methods to compute viewpoints and reconstructs timing from
duration plus `next_start_delta`.

### `ctor.phrase_listener.MidiPhraseListener`

Realtime MIDI helper that records incoming MIDI messages into phrases and can
play phrases back through an output port.

### `ctor.ui.gradio_app.Continuator_gradio`

Local Gradio UI wrapper around `Continuator2`, MIDI ports, phrase memory, piano
roll display, and generation controls.

The older `ctor.continuator_gradio.Continuator_gradio` import remains as a
compatibility wrapper.

## Practical Import Guide

For the current generic model:

```python
from ctor.classic import Variable_order_Markov
from ctor.constraints import ConstraintProblem
```

For the default MIDI Continuator:

```python
from ctor.continuator import Continuator2
```

For the classic MIDI Continuator:

```python
from ctor.classic import ClassicContinuator
```

Compatibility imports such as
`from ctor.variable_order_markov import Variable_order_Markov` and
`from ctor.core import ContextBPModel` remain supported.

For direct sparse chain inference:

```python
from ctor.chain_solver import SparseForwardBackward
```

For future redesign work, treat this current set of classes as the "classic"
implementation to preserve and compare against.

For external compatibility details, see `docs/public_api.md`.
