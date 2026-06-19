# Continuator Cleanup TODO

The default MIDI-facing engine is now `VORegularBPContinuator` through
`ctor.continuator.Continuator2`. The next cleanup phase should reduce the
project around that path and remove older branches of behavior once the default
engine has enough coverage in real use.

## Engine Cleanup

- Keep `ctor.continuator.Continuator2` backed by `VORegularBPContinuator`.
- Audit callers that still use `ClassicContinuator`, `Variable_order_Markov`,
  or `ContextBPContinuator` directly.
- Move any useful tests, examples, or documentation from the classic and
  context-BP engines onto the VO-Regular-BP path.
- Once no important caller depends on them, remove the two older MIDI engines:
  the classic MIDI facade and the context-BP MIDI facade.
- Keep only the generic pieces that are still useful for research or tests. If
  the generic classic `Variable_order_Markov` is no longer needed either, plan
  its removal separately because it still supports non-MIDI examples.

## Decay Modes

- Treat the classic decay-mode controls (`full`, `late`, `middle`, `early`) as
  deprecated.
- Remove the Gradio decay-mode control once the UI question is settled.
- Remove the `set_decay_mode(...)` compatibility shim from the default engine
  after external clients no longer call it.
- Delete the lazy exponential counter machinery if the classic engine is
  removed and no other code uses it.

## Gradio Interface

- Reassess whether maintaining the local Gradio interface is still useful.
- If the maintained front end is now external, deprecate
  `ctor.continuator_gradio` and `ctor.ui.gradio_app`.
- Before removing it, check whether it is still needed for local MIDI testing,
  demos, or quick debugging.
- If it stays, simplify it around the VO-Regular-BP default and remove
  controls that only made sense for the classic engine.

## Compatibility Surface

- Keep the high-level `Continuator2` MIDI methods stable during the transition:
  `learn_phrase`, `sample_sequence`, `continue_sequence`,
  `continue_until_end`, `realize_vp_sequence`, and `save_midi`.
- Prefer removing engine-specific compatibility shims only after docs,
  examples, tests, and external front ends have moved to the default semantics.
- Update `docs/public_api.md` whenever a compatibility method is deprecated or
  removed.
