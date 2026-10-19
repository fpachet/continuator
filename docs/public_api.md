# Continuator public API

## Maintained imports

```python
from ctor.continuator import Continuator2
from ctor.vo_regular_bp import VORegularBPContinuator
from ctor.sequence_model import SequenceModel
from ctor.constraints import ConstraintProblem
from ctor.belief_propag import NoSolutionErrorInBP
from vo_regular_bp import ConstraintSet
```

`Continuator2` remains the stable MIDI entry point. Both MIDI names use the
same backend. `ctor.vo_regular_bp_continuator.VORegularBPContinuator` also
remains a wrapper import. Historical engine removals are listed in
[migration.md](migration.md).

## MIDI facade

```python
Continuator2(midi_file=None, kmax=4, transposition=False,
             policy=None, augmentation_mode="explicit", transposition_offsets=None,
             virtual_transposition=False, seed=0,
             cache_size=8, cache_bytes=64 * 1024 * 1024)

learn_file(midi_file, transposition)
learn_folder(folder_path, transpose=False)
learn_files(files, transposition=False)
learn_phrase(note_sequence, transposition)
learn_phrase_from_mido(messages)
get_phrase_from_mido(messages)

sample_sequence(prefix=None, length=50, constraints=None, start_vp=None,
                relax_prefix_on_fail=True, relax_pos0_on_fail=True, raise_on_fail=False)
continue_sequence(prefix, length=50, constraints=None,
                  relax_prefix_on_fail=True, relax_pos0_on_fail=True, raise_on_fail=False)
continue_until(prefix=None, *, stop, min_length=1, max_length=64,
               constraints=None, raise_on_fail=False)
continue_until_end(prefix=None, min_length=1, max_length=64, end_vp=None,
                   *, raise_on_fail=False)

get_start_vp()
get_end_vp()
get_viewpoint(note)
get_vp_for_pitch(pitch)
realize_vp_sequence(viewpoints)
create_mido_sequence(notes, tempo=120, sustain=False)
save_midi(notes, output_file, tempo=120, sustain=False)

get_phrase_titles()
get_phrase(index)
clear_memory()
clear_last_phrase()
clear_first_n_phrases(n)
set_learn_input(value)
get_learn_input()
set_transpose(value)
set_forget(value)
set_keep_last(n)
get_last_generation_trace()
get_last_generation_diagnostics()
```

Generation returns viewpoints, not notes. Prefixes accept Note objects or
viewpoints. Dictionary and ConstraintProblem values are viewpoints (the MIDI
facade also converts Note values); ConstraintSet uses symbolic viewpoints.
`get_vp_for_pitch` chooses one learned viewpoint for that pitch using the seeded
RNG and raises ValueError if the pitch was never learned.

`set_transpose` controls future learning. Explicit transposition stores shifted
phrases. Virtual augmentation uses one transform set for the symbolic memory;
turning it on during learning transforms the existing base corpus as well.
Use a new model to change an established virtual transform set.

## Generic symbols

```python
model = SequenceModel(kmax=4, seed=0)
model.learn_sequences([["C", "D", "E"], ["C", "F", "E"]])
sequence = model.sample_sequence(3, constraints={0: "C", 2: "E"})
continuation = model.continue_sequence(["C"], length=2)
ending = model.continue_until_end(["C"], min_length=2, max_length=8)
```

SequenceModel accepts optional `viewpoint_fn`, `policy`, `cache_size` and
`cache_bytes`. Its `sample_sequence(length, *, ...)` uses length as the first
argument; the MIDI facade keeps its historical prefix-first signature. Generic
constraints contain symbols/viewpoints, and prefixes may contain original
objects or already learned viewpoints. It exposes `start_symbol`, `end_symbol`,
`order_model`, `last_trace`, `has_viewpoint`, `get_viewpoint`, `clear_memory` and
`get_last_generation_diagnostics`. `learn_sequences` is the batch-learning form.

Learning and `get_viewpoint` always map original objects through viewpoint_fn.
For prefixes, an already learned viewpoint takes precedence. If raw objects
can themselves equal learned viewpoints (for example an integer-to-integer
mapping), explicitly map the prefix with get_viewpoint before passing it.

## Generation contract

- Unprompted fixed generation uses free initial symbols; it can generate a
  learned phrase suffix. Set `start_vp=get_start_vp()` to require a phrase start.
- Explicitly constrained emitted START is supported at position 0. END is
  excluded except when explicitly allowed at the final position. A final
  one-of containing END may choose either END or an allowed note.
- Prefix and hidden handoff are conditioning only; they are not returned.
  If both are supplied, the explicit `start_vp` takes precedence.
- Positions index the generated sequence. ConstraintProblem.length, when set,
  must equal the requested fixed length. Out-of-range positions and invalid
  lengths raise ValueError. Zero fixed length returns an empty list.
- Until generation first hits the stop symbol/set/predicate within the inclusive
  length window; that final stop symbol counts toward length. Omitted prefix
  means hidden START. There is no unbounded infinite computation; clients can
  repeatedly request continuations and pass back the resulting handoff.
- Infeasible or unknown-symbol requests return None. `raise_on_fail=True`
  instead raises NoSolutionErrorInBP. Invalid configuration/type errors remain
  ValueError/TypeError rather than being reported as infeasibility.
- Hard constraints are strict. Legacy `relax_prefix_on_fail` and
  `relax_pos0_on_fail` arguments remain accepted for frontend compatibility but
  never drop a prefix or constraint. Callers implement and report retries.
- Seed controls symbolic draws and MIDI pitch-viewpoint selection. Identical
  input order, options and call order reproduce results for the same backend.
- Learning/forgetting/clearing invalidates inference caches and the last trace.
  Transposition-setting changes also invalidate caches. Public model counts
  should be treated as read-only; mutations must go through learning methods.

`sample_sequence_0` is a deprecated forwarding name for ordinary sampling.
`set_decay_mode` is retained for frontend compatibility: full is accepted,
other settings issue DeprecationWarning and have no effect. The local UI no
longer offers the inactive decay control.

## Optional UI

The existing `python -m ctor.continuator_gradio` launch command and
`ctor.ui.gradio_app.Continuator_gradio` import remain. Install local MIDI/UI
extras when needed. The two modes are fixed length and bounded Until end.
