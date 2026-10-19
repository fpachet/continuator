# Continuator simplification plan

Keep one maintained engine, backed by the external `vo_regular_bp` library,
and preserve `from ctor.continuator import Continuator2` as the MIDI entry
point. Archive the classic and context-BP implementations after closing the
behavior gaps below. This is a proposed migration; the audit has not removed
or moved production code.

## Engine inventory

There are three generation engines, despite the four names in the README:

| Engine | Implementation | Public entry point |
| --- | --- | --- |
| Classic | `ctor/classic/` and `ctor/chain_solver.py` | `ClassicContinuator`, `Variable_order_Markov` |
| Context BP | `ctor/context_bp/` | `ContextBPContinuator`, `ContextBPModel` |
| VO Regular BP | external `vo_regular_bp`, MIDI adapter in `ctor/vo_regular_bp/` | `VORegularBPContinuator` |

`Continuator2` already subclasses `VORegularBPContinuator`. `ctor/core/`,
`ctor/context_bp_continuator.py`, and `ctor/vo_regular_bp_continuator.py` are
compatibility wrappers, not additional engines.

## Audit on 19 October 2026

The initial suite passed all 76 tests. The expanded suite has 110 passing
tests and three strict expected failures documenting VO facade gaps. Both the
project's pinned backend `v0.1.1` and the installed sibling checkout were
tested. The installed distribution reports version `0.1.0`, so source identity
matters more than that metadata:

- Pinned `v0.1.1`: `7ca7bcbce536a23fa8a190af6e59db174f55d49d`.
- Installed clean checkout: `ae05b8799a9698d986afbe2b010194b32aa2c678`.

The audit runs 100 draws for each of eight scenarios per engine: unconstrained
fixed length, final note, internal one-of plus final note, prompted final note,
hidden handoff, fixed END, until END, and prompted until END. All 2,400 draws
succeeded with the requested constraints, lengths, learned first-order
transitions, and realizable pitches. On this branching synthetic corpus,
fixed-length batches had 98–100 distinct outputs per 100 draws; until-END
batches had 70–90. These are checks of variety, not listening-based judgments
of musical quality or proof of a particular sampling distribution.

Each engine also completed 100 successive 16-note continuations, feeding back
realized notes: 1,600 notes and 100 distinct chunks per engine. This is a bounded
streaming smoke test. The local UI offers bounded `Until end`, with a maximum
window of 500, rather than an unbounded infinite generator. Hardware playback
and any infinite mode in an external front end remain outside this audit.

Regression tests additionally cover MIDI serialization and reading, explicit
transposition, clearing/relearning memory, impossible known transitions, and
unreachable END windows. VO-specific tests cover arbitrary stop symbols, stop
sets, stop predicates, and a forbidden motif combined with a final note.

### Correctness and compatibility gaps

1. VO unprompted sampling injects hidden START. After learning `[60, 62, 64]`,
   requesting length 2 ending in END fails, although `[64, END]` is possible
   under the classic and context-BP free-initial semantics.
2. A constraint explicitly requiring emitted START at position 0 fails in VO.
   Classic accepts `[START, 60, 62, 64, END]`. Context-BP also rejects this
   example, so the three facades do not have identical marker semantics.
3. A final one-of containing END is incorrectly routed to END-only generation.
   With the same learned phrase, length 3 ending in `{64, END}` is feasible,
   but VO rejects it. `_final_position_accepts_end()` tests membership rather
   than whether END is the only allowed final symbol.
4. The older engines can leak `KeyError` or `AttributeError` for unknown notes
   or unknown constraint viewpoints. VO returns `None` for the tested default
   calls. Error behavior needs a single explicit contract.
5. VO accepts `relax_prefix_on_fail` and `relax_pos0_on_fail` but does not use
   them. It also accepts decay settings as a no-op. Preserve strict constraint
   guarantees and make any relaxation visible to callers.

### Performance

Measurements use Python 3.13.11 on macOS ARM64, `data/prelude_c.mid` with 549
notes, order 4, no transposition, five repetitions. These are local medians,
not service latency guarantees. Generation includes inference preparation and
trace creation but excludes MIDI realization. Graph caches may be warm across
calls; the facade nevertheless rebuilds backward messages on every call.

| Engine and source | Fixed 64 notes with final-note constraint | Until END, window 4–64 |
| --- | ---: | ---: |
| Classic | 12 ms | 9 ms |
| Context BP | 46 ms | 562 ms |
| VO pinned v0.1.1 | 21 ms | 1,052 ms |
| VO installed checkout | 44 ms | 54 ms |

The pinned VO until-END path prepares a separate backend for every possible
length. Its medians are 261 ms, 1,052 ms, and 4,329 ms for maximum lengths 32,
64, and 128: approximately quadratic growth. The installed checkout shares
first-hit backward messages between lengths; the corresponding times are
28 ms, 54 ms, and 106 ms. Its maximum-500 run takes about 419 ms. The newer
checkout improves this mode substantially but is slower for fixed-length
generation, so changing the dependency deserves a measured release decision.

For pinned VO at 64 notes, preparation takes about 21 ms while repeated sampling
of one prepared backend takes about 0.4–0.5 ms. The facade currently discards
that prepared backend after every request. With 12 explicit transpositions,
generation rises to about 198 ms; virtual transposition takes about 204 ms.
Virtual learning is cheaper here (about 3.5 ms versus 23 ms), but inference
still expands the transformed state space: both modes peak at about 35 MiB
of traced Python allocations, compared with 4.3 MiB without augmentation.
These allocation figures exclude total process/native-library memory.

Other scaling measurements on pinned VO:

- Order 8 versus order 4: about 101 ms versus 21 ms for 64 generated notes.
- Increasing a diverse 12-pitch corpus from 512 to 8,192 notes: generation
  rises from about 24 ms to 158 ms; sequential learning from 6 ms to 1,353 ms.
- Repeating the same Bach phrase 16 times leaves symbolic inference near
  21 ms but increases realization from about 2 ms to 64 ms. These repeated
  phrases increase realization addresses without increasing context support.

The main costs are repeated preparation, whole-memory rebuilding after each
learned phrase, transformed context expansion, and the shared realizer's dense
pairwise address comparisons. Classic's lower timings do not establish
equivalent inference semantics: it uses first-order feasibility with a
variable-order sampler.

Raw measurements and compatibility probes are in
`benchmarks/results/engine_audit_v0_1_1.json` and
`benchmarks/results/engine_audit_local.json`. Reproduce the installed-backend
run from the project root with:

```bash
python -m pytest -q
python -m benchmarks.engine_audit --samples 100 --repeats 5 --long-until \
  --output /tmp/continuator-audit.json
```

For the pinned-source comparison, export the sibling repository's `v0.1.1`
tag into a temporary directory and put that directory first in `PYTHONPATH`.
Omit `--long-until` for that release. MIDI file hashes and backend paths are
recorded in the JSON so different inputs or editable installations are visible.

## Proposed implementation order

### 1. Establish the single-engine contract

- Turn the three expected failures into passing VO regression tests. Define
  free-initial versus phrase-start generation, explicit boundary markers, and
  final note/END unions deliberately.
- Define errors, empty/unknown prefixes, constraint indexing, END counting,
  reproducible RNG seeds, and strict versus relaxed generation in
  `docs/public_api.md`. Never silently discard a requested hard constraint.
- Audit `continuator_front` and other external clients before changing their
  imports or the signature/meaning of compatibility methods.
- Keep the existing engine comparisons until these acceptance criteria pass.

### 2. Make VO efficient at the facade boundary

- Release/pin a tested backend containing the shared first-hit optimization;
  check the fixed-length slowdown before adopting the installed revision.
- Reuse prepared inference for repeated requests. Bound the cache and key it
  by model revision, prefix, horizon, normalized constraints and policy. Clear
  or version it after learning, forgetting, transposition changes and clearing
  memory. Arbitrary predicate callbacks require explicit safe cache handling.
- Add a batch-learning path, or defer symbolic rebuilding until generation,
  so loading a folder does not rebuild all prior phrases after each file.
- Keep virtual augmentation optional; its learning savings do not imply cheap
  generation. Optimize realization separately when repeated addresses dominate.
- Rerun these benchmarks after each change; set latency targets for the actual
  front-end interaction and corpus rather than adding machine-specific CI limits.

### 3. Separate the maintained responsibilities

Use a thin public MIDI facade, one generic symbolic adapter to `vo_regular_bp`,
phrase/realization memory, and the existing shared MIDI utilities. Preserve
the useful separation already present in `ctor/vo_regular_bp/`; do not duplicate
BP or automaton algorithms in this repository.

Migrate integer, character, word and chord examples from the classic model to
the generic adapter before removing the old generic APIs. Remove the two-engine
selection factory once all maintained examples use the one backend. Keep the
Gradio interface and MIDI ports during this migration; remove its ineffective
decay controls and clarify generation-mode labels.

### 4. Archive the two engines and their dependent code

Preserve the pre-removal tree in a named Git archive tag/branch with its tests,
dependency identity and audit results. Keep the historical code out of the
installed wheel and active test discovery. A separate legacy source snapshot
is preferable to retaining another maintained runtime under `ctor/`.

Archive `ctor/classic/`, `ctor/context_bp/`, their comparison examples and
engine-specific tests/docs. Remove `ctor/core/` wrappers and context-BP entry
points with their implementations. Retire `ctor/chain_solver.py` once its
remaining importers/tests have moved with classic. Update the eager classic
import in `ctor/continuator.py` before removing that package.

`Variable_order_Markov` is a separate migration decision: its generic examples,
counter/decay APIs and model inspection are not equivalent to the new backend.
Migrate the supported generation subset to a documented deprecated adapter, or
remove that import in an explicitly breaking release. Do not silently alias it
to a different class. Likewise, explicitly retire `ClassicContinuator` imports;
the stable default `Continuator2` import and high-level MIDI methods remain.

### 5. Finish packaging and documentation

- Make the README describe one active engine, with archive/recovery instructions.
- Update architecture/class maps and public API docs to the final source tree;
  keep historical context-BP design material in the archive.
- Consolidate compatibility wrappers once callers have migrated, and reconcile
  the currently different `ctor.__version__` and project metadata versions.
- Build the distribution in a clean environment with the pinned backend, run
  acceptance tests, verify generic examples and MIDI file output, and exercise
  the external front-end contract. Then repeat the performance audit.

The archive step follows correctness and performance work so the older engines
remain available as references while their useful behaviors are migrated.
