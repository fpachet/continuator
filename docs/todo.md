# Continuator simplification results

The migration is implemented. The active project has one generation engine,
VO-Regular-BP, behind the stable Continuator2 MIDI facade and the new generic
SequenceModel. The classic/context-BP source and comparison audit are preserved
in `codex/archive-three-engines-2026-10-19`; recovery instructions are in
[migration.md](migration.md).

## Completed work

- Fixed all three audited VO gaps: free-initial generation, emitted START at
  position zero, and a final note/END union. The regressions now pass normally.
- Added a generic symbolic adapter and migrated integer, character, word and
  chord examples. Variable_order_Markov remains a warned generation-only
  adapter, with classic-specific APIs explicitly retired in version 2.0.
- Removed the two obsolete implementations, chain solver, engine selector,
  context-BP wrappers, comparison examples and engine-specific tests from the
  maintained package. The archive includes their full runnable source.
- Pinned backend commit `ae05b8799a9698d986afbe2b010194b32aa2c678`, including
  shared first-hit inference. Its 169 upstream tests pass. A fresh remote fetch
  before the migration commit confirmed this is still the latest backend main
  revision; no newer dependency update was available.
- Added reusable prefix-independent fixed plans, prefix-aware until backends,
  bounded LRU admission and lazy diagnostics. Mutable constraint containers are
  snapshotted; callbacks and regular products bypass caching. Learning,
  deleting, clearing and transposition-setting changes invalidate inference.
- Deferred symbolic rebuilding until generation/inspection. Folder learning
  and generic learn_sequences therefore avoid rebuilding prior phrases after
  every input.
- Reduced equivalent realization domains for exact duplicate note phrases,
  preserving separate memory entries, counts, note attributes and timing.
- Kept Gradio/realtime MIDI support, removed the inactive decay control, and
  bounded the default chord example to 256 phrases. Full-corpus learning is an
  explicit `--limit 0` option.
- Updated architecture/API/migration documentation, agent orientation, release
  metadata and packaging. The source distribution explicitly includes the
  tracked Bach MIDI fixture despite the general MIDI ignore rule.

## Verification

The active suite has 71 passing tests and no skips or expected failures in the
local development environment. It covers fixed/variable lengths, one-of and
regular constraints, markers, unknown/infeasible requests, RNG reproducibility,
cache reuse/invalidation/limits, lazy learning, repeated viewpoint mapping,
memory deletion, transposition,
duplicate realization domains and MIDI file roundtrips.

The optional sibling-frontend contract test exercises its default import
fallback and explicit VO selection, memory generation, hidden handoff,
constrained ending, traces, graph rendering and 20 successive continuation
requests using returned handoff viewpoints. The sibling repository was not
modified or deployed. Its explicit context-BP menu choice is retired and its
published dependency pin still needs a separate frontend release update.

The final wheel is checked in a fresh virtual environment using its actual
pinned dependency, outside this source checkout. The optional sibling frontend
check is skipped there because that checkout is absent; the remaining tests
pass. The wheel contains no classic/context-BP/core packages or chain solver.
The source distribution includes tests, benchmarks and example datasets.

All integer, character, word, chord, note and constrained-note examples were
run successfully. MIDI output was generated and decoded. Hardware playback and
the external hosted application were not exercised.

## Performance

Raw evidence is in `benchmarks/results/engine_audit_single_engine.json`.
Historical baselines remain in `engine_audit_v0_1_1.json` and
`engine_audit_local.json` in the same directory. The complete historical
three-engine driver is preserved in the archive; the active driver targets VO.

The final workload makes 800 generation requests across eight scenarios, with
no constraint failures. Fixed-length batches produce 99–100 distinct sequences
per 100 draws on the branching synthetic corpus; until-END batches produce
81–87. A separate rolling check produces 1,600 notes in 100 distinct chunks.
These are variety/contract checks, not subjective musical-quality scores.

Local results on Python 3.13/macOS ARM64, order 4, 549-note Bach corpus,
seven repetitions:

| Request | Before migration, installed backend | New inference plan | Repeated cached request |
| --- | ---: | ---: | ---: |
| 64 notes with constrained final note | 44 ms | 44 ms | 0.63 ms |
| Until END, window 4–64 | 54 ms | 51 ms | 0.39 ms |
| Until END, window 4–500 | 419 ms | 371 ms | 5.3 ms |

The older pinned v0.1.1 baseline took about 1,052 ms for the 4–64 until-END
window and 4,329 ms for 4–128. Shared first-hit tables remove that quadratic
repetition. At the selected backend revision, cold fixed-length preparation is
slower than v0.1.1's approximately 21 ms; caching pays off across repeated
requests and changing prefixes. First use after learning additionally performs
the deferred symbolic-model build.

With 12 transpositions, the repeated constrained 64-note request takes about
0.86 ms explicitly and 0.95 ms virtually once admitted to the cache. First
use is still expensive: about 468 ms and 712 ms including deferred model/graph
work. Virtual augmentation is optional and does not eliminate the transformed
inference state space.

Sixteen identical Bach phrases previously cost roughly 64 ms to realize 64
notes; canonical realization domains reduce this to about 3.1 ms. Counts still
include all sixteen learned phrases.

Lazy ingestion of an 8,192-note diverse corpus takes about 10 ms to queue/index,
then about 288 ms for the first generation including model construction and
inference. Repeated generation is about 1.2 ms. The earlier v0.1.1 path spent
about 1,353 ms in sequential learning alone, plus 158 ms per generation. The
queued-learning number must not be interpreted as total training cost.

## Operational limits

- Cache admission has an eight-entry default and an estimated 64 MiB table
  budget. This is not a total process-memory limit. Backend graph/candidate
  caches and native allocations also consume memory.
- Callback/regular-product requests and constrained until requests prepare
  fresh inference. Large models, long windows and transformed corpora remain
  expensive on their first request.
- Until generation is bounded; streaming is repeated finite continuation.
  Live audio/MIDI latency and hardware behavior require an integration run.
- The recovery tag accompanies this migration. Fetch it before creating a
  recovery checkout on another machine. Frontend deployment and upstream
  backend publication are separate release actions.

Reproduce from the project root:

```bash
python -m pytest -q
python -m benchmarks.engine_audit --samples 100 --repeats 7 --long-until \
  --output /tmp/continuator-audit.json
```

First-call timing and repeated-call medians are reported separately. Non-finite
log-mass diagnostics are stored as strings so the new result file is valid JSON.
