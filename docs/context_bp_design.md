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

## First Implementation

The first implementation lives under `ctor.core`:

- `Vocabulary`: maps arbitrary symbols to integer ids.
- `ContextCounts`: stores variable-order continuation counts.
- `ContextGraph`: compiles counts into sparse context-state edges.
- `forward_backward`: runs exact finite-chain inference on the context graph.
- `ContextBPModel`: user-facing generic model with `learn_sequence`,
  `infer`, `symbol_marginals`, and `sample_sequence`.

This first pass supports fixed-length generation with positional hard
constraints. MIDI integration and until-end generation are intentionally left
for later once the generic semantics are stable.

## Compatibility

The classic API remains unchanged. In particular:

```python
from ctor.continuator import Continuator2
from ctor.variable_order_markov import Variable_order_Markov
```

continue to refer to the classic implementation.
