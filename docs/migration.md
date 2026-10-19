# Migrating to the single engine

Continuator 2.0 maintains only VO-Regular-BP. The former classic and context-BP
implementations, their tests, design notes and the full comparison audit are
preserved in the Git tag `codex/archive-three-engines-2026-10-19`, pointing
to snapshot commit `b929473c72d0e0215f830a71dbc6942334417e8b`. That snapshot includes
previously uncommitted audit files. Creating it did not change the current
branch or index. Fetch the recovery tag when using a different checkout.

## Recover the historical project

Create a separate checkout without altering the maintained working directory:

```bash
git fetch origin tag codex/archive-three-engines-2026-10-19
git worktree add --detach /tmp/continuator-three-engines codex/archive-three-engines-2026-10-19
cd /tmp/continuator-three-engines
python -m pytest -q
```

The snapshot's initial audit suite has 110 passes and three expected failures
on the backend revisions documented there. Its benchmark driver compares all
three engines. The maintained project keeps the original JSON measurements as
historical evidence and a new driver that tests only the maintained engine.

## Public imports

Keep `from ctor.continuator import Continuator2` for MIDI generation. New
symbolic code uses `from ctor.sequence_model import SequenceModel`.

Removed APIs include `ClassicContinuator`, `ctor.classic`, `ctor.context_bp`,
`ctor.context_bp_continuator`, `ctor.core`, `ctor.chain_solver` and the old
`ctor.engines` selection factory. These are intentional 2.0 removals; they are
not aliases that pretend the new backend has the same inference semantics.

`from ctor.variable_order_markov import Variable_order_Markov` remains a warned,
generation-only adapter. Its familiar constructor, learning, viewpoints,
START/END properties and constrained generation use SequenceModel. Classic
counter/decay APIs, first-order matrices, chain marginals, zero-order sampling
and realization-address internals are retired. Use the archived project if
those research APIs are required; there is no equivalent silent replacement.

Migrate generic code explicitly:

```python
# Previous: Variable_order_Markov(sequence, viewpoint_fn, kmax=4)
model = SequenceModel(kmax=4, viewpoint_fn=viewpoint_fn, seed=0)
model.learn_sequence(sequence)
```

Unprompted sampling is free-initial. Explicit START handoffs request a phrase
beginning. Hard constraints are never silently relaxed. Read
[public_api.md](public_api.md) before relying on compatibility keywords.

## External frontend

The sibling `continuator_front` adapter was checked without modifying that
repository. Its classic import falls back to Continuator2 when classic is
absent, so its current default works with this package. Its VO selection also
works. Contract tests cover memory learning/generation, hidden handoff,
constrained ending, traces and graph rendering. The explicit context-BP choice
is unavailable and should be removed by a separate frontend change before its
next release. Its published dependency pin still selects an older Continuator;
updating/deploying that frontend is a separate release action.

Legacy decay settings are ignored with a deprecation warning. The local Gradio
UI no longer advertises them. The maintained package retains optional local
MIDI support.

## Backend pin

The backend is pinned to commit `ae05b8799a9698d986afbe2b010194b32aa2c678` rather
than an editable checkout or moving branch. That commit contains shared
first-hit inference; the previous v0.1.1 implementation prepared every candidate
length separately. Fixed-length cold inference is slower at this backend
revision, but prefix-independent cached plans make repeated requests inexpensive.
No algorithm was copied into this repository and no upstream release was
published as part of this migration.
