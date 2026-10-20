# Continuator

Continuator learns sequences online and generates varied continuations with
positional and regular constraints. A note/viewpoint representation preserves
learned rhythmic and overlap information without tokenizing every timing detail.

There is one maintained engine: the external `vo_regular_bp` order-stack
library. `Continuator2` is the MIDI entry point; `SequenceModel` provides generic
integer, text, chord and other symbolic generation. The default policy backs
off from singleton contexts to encourage variety while respecting constraints.

## Installation

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Core dependencies are NumPy, mido, and an immutable Git revision of
vo-regular-bp. Optional integrations use extras:

```bash
python -m pip install '.[dev]'           # tests
python -m pip install '.[local-ui]'      # Gradio and realtime MIDI ports
python -m ctor.ui.gradio_app
```

Set `GRADIO_SERVER_PORT=8060` if the usual port is occupied.

## MIDI generation

```python
from ctor.continuator import Continuator2

continuator = Continuator2('data/prelude_c.mid', kmax=4, seed=0)
target = continuator.get_vp_for_pitch(60)
viewpoints = continuator.sample_sequence(length=16, constraints={15: target}, raise_on_fail=True)
notes = continuator.realize_vp_sequence(viewpoints)
continuator.save_midi(notes, '/tmp/continuation.mid', tempo=-1)
```

A viewpoint is `(pitch, duration_bin, overlaps_left, overlaps_right)`. A last-note
constraint fixes a complete viewpoint. The external ConstraintSet also accepts
positional predicates when a pitch-only condition is desired.

Generation modes share the same backend:

```python
viewpoints = continuator.sample_sequence(length=16)
viewpoints = continuator.continue_sequence(prefix=played_notes, length=16)
viewpoints = continuator.continue_until_end(prefix=played_notes, min_length=4, max_length=64)
viewpoints = continuator.continue_until(prefix=played_notes, stop=target, max_length=64)
```

Prefixes condition generation and are excluded from output. Until methods return
the first reached stop symbol, including END, within a bounded window. Realizing
a sequence filters out boundary markers. Continuous clients can request successive
chunks and pass back the last realized notes or handoff viewpoint.

Unprompted fixed sampling can start anywhere in learned memory. To require a
phrase beginning, pass `start_vp=continuator.get_start_vp()`. Positional indices
always refer to the returned generated sequence. Infeasible requests return None
unless `raise_on_fail=True`; hard constraints are never silently dropped.

For one-of constraints:

```python
from ctor.constraints import ConstraintProblem

problem = ConstraintProblem(length=16)
problem.at(7).one_of([viewpoint_a, viewpoint_b])
problem.at(15).equals(target)
viewpoints = continuator.sample_sequence(length=16, constraints=problem)
```

`vo_regular_bp.ConstraintSet` adds forbidden motifs and regular constraints.
Explicit and optional virtual transposition remain available through
VORegularBPContinuator/Continuator2. Virtual learning saves base-note memory,
while constrained inference still depends on the transformed state space.

## Generic symbols

```python
from ctor.sequence_model import SequenceModel

model = SequenceModel(kmax=4, seed=0)
model.learn_sequences([[1, 2, 3], [1, 2, 4]])
sequence = model.sample_sequence(3, constraints={0: 1, 2: 4})
continuation = model.continue_sequence([1, 2], length=1)
ending = model.continue_until_end([1], min_length=2, max_length=8)
```

Examples can be run from the repository root:

```bash
python -m examples.int_continuator
python -m examples.char_continuator
python -m examples.word_continuator
python -m examples.chordseq_continuator
python -m examples.constrained_note_continuator
```

The chord demo defaults to 256 training phrases. `--limit 0` explicitly loads
all 31,032 phrases (about 2.5 million chord tokens), which requires substantially
more inference time and memory.

## Performance and validation

Learning invalidates the model and defers its rebuild until generation, so
folder/batch learning avoids repeatedly rebuilding prior material. Repeated
requests reuse prepared inference. Fixed-length backward tables can be reused
across different prefixes; callback/regular-product requests bypass the cache.
Exact duplicate note phrases share equivalent MIDI realization domains while
retaining their separate phrase entries and statistical weights.

```bash
python -m pytest -q
python -m benchmarks.engine_audit --samples 100 --repeats 5 --long-until --output /tmp/audit.json
```

The benchmark records first-call and repeated-call latency, variety, constraints,
rolling continuations, transposition and corpus scaling. Its thresholds are not
machine-specific CI assertions. See [docs/todo.md](docs/todo.md) for measured
migration results and [docs/current_architecture.md](docs/current_architecture.md)
for cache admission and memory limitations.

## Compatibility and archive

Version 2.0 removes the active classic and context-BP engines. Their full source,
tests and comparison audit remain recoverable through the Git tag
`codex/archive-three-engines-2026-10-19`. The stable Continuator2 import remains.
Obsolete launcher/import wrappers and the deprecated Variable_order_Markov
adapter have been removed. Use `SequenceModel` for generic symbols and
`python -m ctor.ui.gradio_app` for the optional local UI. Local MIDI
recording and playback live in `ctor.ui.phrase_listener`.

Read [docs/migration.md](docs/migration.md) for recovery commands and external
frontend notes, [docs/public_api.md](docs/public_api.md) for precise semantics,
and [docs/class_map.md](docs/class_map.md) for the maintained classes. Local UI
and realtime MIDI support remain optional. The external frontend is available
at [continuator_front](https://huggingface.co/spaces/pachet/continuator_front).

## Background

The project builds on François Pachet's Continuator (Journal of New Music
Research, 2003), finite-length constrained Markov processes (IJCAI 2011), and
exact sampling for regular and Markov constraints with belief propagation
(CP 2015). Author: [François Pachet](https://github.com/fpachet).

Licensed under the MIT License; see [LICENSE](LICENSE).
