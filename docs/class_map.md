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
| `ctor.midi.MidiContinuatorBase` | MIDI parsing/writing, note timing and coherent realization. |
| `ctor.midi.MidiRealizationStore` | Notes, source addresses and duplicate-phrase representatives. |
| `ctor.vo_regular_bp.realization_store.TransformAwareMidiRealizationStore` | Transform-indexed addresses and lazy note transforms. |
| `midi_stuff.mini_muse.Note` | Pitch, velocity, duration, start time and overlap/timing metadata. |
| `ctor.ui.gradio_app.Continuator_gradio` | Optional local UI. |
| `ctor.phrase_listener.MidiPhraseListener` | Optional realtime MIDI recording/playback. |
| `ctor.belief_propag.NoSolutionErrorInBP` | Retained public exception name for infeasible generation. |
| `ctor.variable_order_markov.Variable_order_Markov` | Deprecated generation-only SequenceModel adapter; no classic counters or decay. |

There is one engine. The legacy `ctor.vo_regular_bp_continuator` and
`ctor.continuator_gradio` files only preserve entry-point imports. See
[public_api.md](public_api.md) for behavior and [migration.md](migration.md)
for retired classes and archive recovery.
