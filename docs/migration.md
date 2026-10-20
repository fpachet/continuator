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

The deprecated `ctor.variable_order_markov.Variable_order_Markov` adapter
has now also been removed. Use `SequenceModel` directly. Classic counter/decay
APIs, first-order matrices, chain marginals, zero-order sampling and
realization-address internals remain retired. Use the archived project if
those research APIs are required; there is no equivalent silent replacement.

Migrate generic code explicitly:

```python
# Previous: Variable_order_Markov(sequence, viewpoint_fn, kmax=4)
model = SequenceModel(kmax=4, viewpoint_fn=viewpoint_fn, seed=0)
model.learn_sequence(sequence)
```

Unprompted sampling is free-initial. Explicit START handoffs request a phrase
beginning. Hard constraints are never silently relaxed. Read
[public_api.md](public_api.md) for maintained generation arguments.

## Removed wrappers and local UI

The following historical paths have been removed. Their replacements use
the same maintained engine or local UI implementation:

| Removed path | Maintained replacement |
| --- | --- |
| `ctor.continuator_gradio` | `python -m ctor.ui.gradio_app` |
| `ctor.midi_ports_poll` | `python -m ctor.ui.midi_ports_poll` |
| `ctor.vo_regular_bp_continuator` | `from ctor.vo_regular_bp import VORegularBPContinuator` |
| `ctor.variable_order_markov` | `from ctor.sequence_model import SequenceModel` |
| `ctor.phrase_listener` | `from ctor.ui.phrase_listener import MidiPhraseListener` |

`ctor.continuator.Continuator2`, `ctor.constraints.ConstraintProblem` and
`ctor.belief_propag.NoSolutionErrorInBP` remain maintained public imports.

## External frontend

The sibling `continuator_front` now uses only `Continuator2`; its engine
selector and retired decay controls have been removed. The contract test
covers memory learning/generation, prefix conditioning, hidden handoff,
constrained ending, traces, graph rendering and successive continuations.
The frontend pins an immutable Continuator commit for deployment; after
updating that pin, run its backend integration tests against that revision.

Unused `sample_sequence_0`, `set_decay_mode`, `relax_prefix_on_fail` and
`relax_pos0_on_fail` APIs have now been removed. Use ordinary sampling with
strict constraints; the sibling frontend already uses that contract. Unused
legacy constraint conversion/shifting helpers, Note comparison helpers and the
`utils.profiler` package are also removed. The maintained package retains
optional local MIDI support.

## Backend pin

The backend is pinned to commit `5e8526473af13ee1106d616410284fa01f2c6d54`. This builds on
`ae05b8799a9698d986afbe2b010194b32aa2c678`'s shared first-hit inference and
optimizes cold positional inference with repeated-mask transition rows and a
single numeric pass for underflow detection. BP and automaton code remain in
the external dependency. This optimization was merged through backend PR #1.
A newer upstream snapshot-update API is available but is not yet used here;
see [the integration roadmap](todo.md#next-step-integrate-online-snapshots)
for its measured benefit, implementation plan and validation requirements.
