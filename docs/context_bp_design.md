# Context-BP Core Design

The context-BP engine is a new generic core built beside the classic
`Variable_order_Markov` implementation. It is intentionally separate while the
semantics are tested.

## Central Idea

A variable-order Markov model can be made first-order by lifting the state from
"current symbol" to "current context".

For a context such as:

```text
(A, B)
```

an emitted next symbol `C` creates an edge:

```text
(A, B) -- emit C --> (B, C)
```

with the transition weight learned from the longest available suffix of the
current context. Positional constraints apply to the emitted symbol labels on
these edges.

For constrained sampling, the engine uses stepwise order backoff. It computes
backward feasibility messages for each order from `kmax` down to 1. At each
emitted position, the sampler builds the feasible candidate sets from longest
to shortest context and delegates the final choice to an explicit order policy.

The generic `ContextBPModel` defaults to `LongestFeasiblePolicy`: use the
longest context that can still complete the remaining constraints. This keeps
the generic core close to exact context-graph BP semantics.

`SingletonAvoidingBackoffPolicy` reproduces the practical classic Continuator
heuristic. After BP/constraint filtering, a higher-order context with a single
continuation is usually skipped, and that skipped symbol is suppressed at
intermediate lower orders when alternatives exist. Order 1 remains the fallback
so generation can still proceed when the singleton is forced. The MIDI-facing
`ContextBPContinuator` uses this policy by default.

`infer(...)` and `symbol_marginals(...)` still return exact BP results on one
effective context graph: they try `kmax`, then `kmax - 1`, down to order 1,
and return the highest globally feasible order.

`sample_sequence_with_trace(...)` returns the generated sequence plus one
`SampleStep` per emitted symbol. Each step records both the globally feasible
graph order selected at that step and the actual suffix order that supplied the
chosen transition. This is useful for diagnosing where the model is backing off.

## First Implementation

The implementation lives under `ctor.context_bp`:

- `Vocabulary`: maps arbitrary symbols to integer ids.
- `ContextCounts`: stores variable-order continuation counts.
- `ContextGraph`: compiles counts into sparse context-state edges.
- `forward_backward`: runs exact finite-chain inference on the context graph.
- `LongestFeasiblePolicy` and `SingletonAvoidingBackoffPolicy`: choose among
  per-step feasible context orders during sampling.
- `ContextBPModel`: user-facing generic model with `learn_sequence`,
  `infer`, `symbol_marginals`, `sample_sequence`, and
  `continue_until_end`.

This first pass supports fixed-length generation with positional hard
constraints and variable-length first-hit generation to an end symbol. MIDI
integration is intentionally left for later once the generic semantics are
stable.

## Engine Selection

The classic and context-BP generic engines can be selected explicitly through
`ctor.engines`:

```python
from ctor.engines import make_sequence_engine

classic = make_sequence_engine("classic", kmax=4)
context_bp = make_sequence_engine("context_bp", kmax=4)
```

This is deliberately not wired into `Continuator2` yet. The current MIDI-facing
API stays classic while the new core is compared on generic sequences.

For MIDI experiments, use the separate `ContextBPContinuator` class:

```python
from ctor.context_bp import ContextBPContinuator
```

It uses `ContextBPModel` for viewpoint generation and keeps a classic
`Variable_order_Markov` store for note realization. It does not inherit from
`Continuator2`; both MIDI facades can use model-agnostic utilities from
`ctor.midi`.

## Until-End Generation

`ContextBPModel.continue_until_end(...)` uses first-hit semantics: the target
symbol is forbidden before the final generated position and forced at the final
position. Feasible lengths inside the requested window are weighted by their
exact path mass at the highest feasible effective order, then a fixed-length
context-BP sample is drawn for the chosen length.

## Compatibility

The classic API remains unchanged. In particular:

```python
from ctor.continuator import Continuator2
from ctor.variable_order_markov import Variable_order_Markov
```

continue to refer to the classic implementation.

The old `ctor.core` import path remains as a compatibility wrapper for
`ctor.context_bp`.
