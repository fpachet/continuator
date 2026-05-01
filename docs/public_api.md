# Public API Compatibility

This project is used by external front ends, including `continuator_front`.
The imports and methods in this document should be treated as compatibility
surface for the current classic engine.

## Stable Imports

These compatibility imports should continue to work:

```python
from ctor.continuator import ClassicContinuator
from ctor.continuator import Continuator2
from ctor.variable_order_markov import Variable_order_Markov
from ctor.constraints import ConstraintProblem
```

New code may prefer the explicit package locations:

```python
from ctor.classic import ClassicContinuator, Continuator2, Variable_order_Markov
from ctor.context_bp import ContextBPModel, ContextBPContinuator
from ctor.context_bp import LongestFeasiblePolicy, SingletonAvoidingBackoffPolicy
from ctor.constraints import ConstraintProblem
```

These legacy/compatibility imports should also keep working, even if their
implementation lives elsewhere:

```python
from ctor.continuator_gradio import Continuator_gradio
from ctor.chain_solver import SparseForwardBackward
from ctor.belief_propag import NoSolutionErrorInBP
from ctor.dynaprog import VariableDomainSequenceOptimizer
from ctor.markov_analysis import MarkovAnalysis, analyze_markov_chain
```

The experimental generic engine-selection API is available separately:

```python
from ctor.engines import make_sequence_engine
```

The experimental MIDI-facing context-BP class is also separate:

```python
from ctor.context_bp import ContextBPContinuator
```

It is no longer a subclass of `Continuator2`. Neither one changes the existing
`Continuator2` compatibility surface.

`ContextBPContinuator` additionally exposes trace helpers for experimental
front ends:

```python
sample_sequence_with_trace(...)
continue_until_end_with_trace(...)
get_last_generation_trace()
```

## `Continuator2`

`Continuator2` is the public MIDI-facing facade. It should remain importable
from `ctor.continuator`. It is now a compatibility subclass of
`ClassicContinuator`, which is the explicit name for the classic MIDI engine.

Constructor:

```python
Continuator2(midi_file=None, kmax=4, transposition=False)
```

Important public methods:

```python
learn_file(midi_file, transposition)
learn_folder(folder_path, transpose=False)
learn_files(files, transposition=False)
learn_phrase(note_sequence, transposition)
learn_phrase_from_mido(phrase)
get_phrase_from_mido(phrase)

sample_sequence(
    prefix=None,
    length=50,
    constraints=None,
    start_vp=None,
    relax_prefix_on_fail=True,
    relax_pos0_on_fail=True,
    raise_on_fail=False,
)

continue_sequence(
    prefix,
    length=50,
    constraints=None,
    relax_prefix_on_fail=True,
    relax_pos0_on_fail=True,
    raise_on_fail=False,
)

continue_until_end(prefix=None, min_length=1, max_length=64, end_vp=None)
sample_sequence_0(length=50, constraints=None)

realize_vp_sequence(vp_seq)
create_mido_sequence(sequence, tempo=120, sustain=False)
save_midi(sequence, output_file, tempo=120, sustain=False)

get_start_vp()
get_end_vp()
get_vp_for_pitch(pitch)
get_viewpoint(note)

set_learn_input(value)
get_learn_input()
set_transpose(trans)
set_decay_mode(choice)
set_forget(forget_past)
set_keep_last(keep)
clear_memory()
clear_last_phrase()
clear_first_n_phrases(n)
get_phrase_titles()
get_phrase(index)
```

## Generation Semantics

The following semantics are compatibility-sensitive:

- `sample_sequence(..., start_vp=...)` treats `start_vp` as a hidden handoff
  viewpoint and does not include it in the returned sequence.
- `continue_sequence(prefix, ...)` uses the prefix only as conditioning context.
  The returned sequence contains only generated material.
- `continue_until_end(prefix, ...)` also excludes the prefix and includes the
  first reached end viewpoint when generation succeeds.
- Positional constraints are indexed over the returned generated sequence, not
  over the prefix plus generated sequence.
- Legacy dict constraints and `ConstraintProblem` are both accepted.
- Public methods return `None` on unsatisfied constraints unless
  `raise_on_fail=True` requests an exception.

## UI Modules

The Gradio UI implementation now lives in:

```python
from ctor.ui.gradio_app import Continuator_gradio
```

The older import remains supported:

```python
from ctor.continuator_gradio import Continuator_gradio
```

The old launch command remains supported:

```bash
python -m ctor.continuator_gradio
```

New code may prefer:

```bash
python -m ctor.ui.gradio_app
```

If the default Gradio ports are already in use, set an explicit port:

```bash
GRADIO_SERVER_PORT=8060 python -m ctor.continuator_gradio
```

## Legacy Helpers

Experimental helpers that are not part of the core runtime live under
`ctor.legacy`, with old imports preserved as wrappers:

```python
from ctor.legacy.dynaprog import VariableDomainSequenceOptimizer
from ctor.legacy.markov_analysis import MarkovAnalysis, analyze_markov_chain
```

These helpers should not constrain the future context-BP core design.
