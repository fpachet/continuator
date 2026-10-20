# Continuator class map

| Class | Responsibility |
| --- | --- |
| `ctor.continuator.Continuator2` | Stable MIDI entry point; subclasses VORegularBPContinuator. |
| `ctor.vo_regular_bp.VORegularBPContinuator` | Note/viewpoint facade, phrase learning, explicit/virtual transposition. |
| `ctor.sequence_model.SequenceModel` | Generic symbols, lazy model, constrained generation, RNG and prepared inference cache. |
| `ctor.sequence_model.Boundary` | Generic START/END markers. |
| External `vo_regular_bp.OrderStackModel` | Variable-order transition counts and context graphs. |
| External `vo_regular_bp.VirtualAugmentedOrderStackModel` | Symbolic counts under optional virtual transforms. |
| External `SingletonAvoidingBackoffPolicy` | Default practical order/backoff policy. |
| `ctor.constraints.ConstraintProblem` | Equality and one-of positional constraints. |
| External `vo_regular_bp.ConstraintSet` | Positional and regular constraints, including forbidden motifs. |
| `ctor.midi.MidiContinuatorBase` | MIDI import/export, note timing and coherent realization. |
| `ctor.midi.MidiRealizationStore` | Notes, source addresses and duplicate-phrase representatives. |
| `ctor.vo_regular_bp.realization_store.TransformAwareMidiRealizationStore` | Transform-indexed addresses and lazy note transforms. |
| `midi_stuff.mini_muse.Note` | Pitch, velocity, duration, start time and overlap/timing metadata. |
| `ctor.ui.gradio_app.Continuator_gradio` | Optional local UI. |
| `ctor.ui.phrase_listener.MidiPhraseListener` | Optional realtime MIDI recording/playback. |
| `ctor.belief_propag.NoSolutionErrorInBP` | Retained public exception name for infeasible generation. |

There is one engine. Obsolete import/launcher wrappers and the deprecated
Markov adapter have been removed. `Continuator2` remains the stable frontend
entry point; `NoSolutionErrorInBP` remains the public infeasibility exception.
See [public_api.md](public_api.md) for behavior and [migration.md](migration.md)
for replacement imports and archive recovery.
