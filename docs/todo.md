# Continuator simplification results

The maintained project has one engine: external `vo_regular_bp`, accessed through
`Continuator2` for MIDI and `SequenceModel` for generic symbols. Historical classic
and context-BP code and comparison audits remain recoverable from the archive tag
in [migration.md](migration.md).

## Completed follow-up cleanup, October 20

- Shared non-mutating MIDI parsing now merges synchronous tracks, matches notes
  by channel/pitch, handles overlapping repetitions with FIFO pairing and returns
  onset order. Incomplete notes and unmatched offs are ignored. File times use
  musical beats; realtime seconds are normalized to 120 BPM. See the explicit
  timing contract in [public_api.md](public_api.md).
- MIDI export honors the requested BPM and rounds absolute ticks before delta
  conversion. Hardware output skips metadata; recording callbacks copy messages.
  Type 2 MIDI files are rejected because their tracks have independent timelines.
- Removed unused legacy constraint converters, profiler package, Note comparison
  helpers, duplicate memory deletion methods, retired decay/zero-order forwarding
  methods and ignored relaxation keywords. The sibling frontend no longer uses
  those controls. The stable `ctor.continuator.Continuator2` import is preserved.
- The MIDI facade owns phrase deletion/reindexing and invalidates the symbolic
  owner. Generation traces and diagnostics have one owner, SequenceModel; the
  facade formats traces on demand. Lazy learning, bounded inference caches,
  virtual transposition and canonical duplicate realization domains remain.
- Published the upstream inference improvement as
  [vo-regular-bp PR #1](https://github.com/fpachet/vo-regular-bp/pull/1), pinning
  immutable commit `5e8526473af13ee1106d616410284fa01f2c6d54`. Repeated positional
  masks share numeric transition rows during preparation, with at most eight
  cached masks. Underflow detection uses the numeric pass, retaining log-space
  fallback. No BP/automaton implementation or extra dependency was added here.
- Added CI for Python 3.11 and 3.13: source tests, wheel/sdist builds, isolated
  installed-wheel tests and a smaller audit with JSON artifacts. The distribution
  checker copies only tests/data/benchmarks into its working directory, so source
  packages cannot shadow the installed wheel. The audit now fails on unexpected
  feasibility/probe errors and separately measures requests after learning.

## Verification

The local source suite passes **84 tests**, including the real sibling frontend's
single-engine contract, handoff, END, graph rendering and successive continuations.
The upstream suite passes **178 tests**. New upstream coverage compares messages
against exhaustive path enumeration, checks forbidden-symbol overrides and
preserves finite log mass for a single edge product that underflows to zero.

Fresh Python 3.11 and 3.13 environments installed the actual wheel and pinned
backend from GitHub: **83 tests pass, one skips** in each because the optional
sibling frontend is absent.
The wheel contains no retired engine or profiler packages. The source distribution
contains the MIDI fixture, tests, benchmarks and distribution checker.

The required full workload uses 100 samples, five repeats and long until windows.
All **800 generation requests** satisfy their constraints, all eight contract
probes pass, and rolling continuations produce **1,600 notes in 100 distinct
chunks**. Fixed-length scenarios have 99–100 distinct outputs per 100 draws;
until scenarios have 81–87. The before/after runs retain the same seeded
acceptance outputs. These checks measure variety and correctness, not subjective
musical quality. Physical MIDI hardware and hosted deployment were not exercised.

## Measured performance

The following compares the previous backend `ae05b879...` with the optimized pin,
using identical final Continuator source and MIDI parsing, Python 3.13/macOS ARM64,
549 Bach notes, order four, and five repetitions. There are no CI timing thresholds.

| Request | Previous backend | Optimized backend |
| --- | ---: | ---: |
| Fresh fixed plan, 64 notes and constrained final note, median | 43.7 ms | 21.7 ms |
| First 64-note request after learning 16 notes, median including rebuild | 53.0 ms | 29.6 ms |
| Repeated cached 64-note request, median | 0.67 ms | 0.69 ms |
| Until END, window 4–64, first preparation/sample | 50.0 ms | 22.3 ms |
| Until END, window 4–500, first preparation/sample | 370.8 ms | 166.8 ms |

Raw reports: [previous backend](../benchmarks/results/engine_audit_cleanup_baseline.json)
and [optimized backend](../benchmarks/results/engine_audit_cleanup_optimized.json).
They include backend source hashes, commit identifiers, model sizes, allocation
measurements and separate first/repeated timings. Earlier migration and three-engine
measurements remain in the other `benchmarks/results/engine_audit_*.json` files.

## Next step: integrate online snapshots

**Status: upstream implementation is available; Continuator integration is pending.**
Backend commit
[`66627cb0f1747bf8a4deb6af146e3a9a5d47a3fe`](https://github.com/fpachet/vo-regular-bp/commit/66627cb0f1747bf8a4deb6af146e3a9a5d47a3fe)
adds the immutable snapshot API:

```python
model = model.with_sequences([new_symbolic_phrase])
```

It counts only added independent phrases and reuses unaffected ordinary source
graph rows. Backward messages and sampling candidates are recomputed. Updates
can fall back to full graph construction when the topology requires it. Virtual
augmentation updates base counts but rebuilds transformed graphs. Fully
incremental BP was deferred because measured probability changes propagate widely
through the backward tables; substantial virtual-graph reuse was also deferred.
See the [upstream investigation](https://github.com/fpachet/vo-regular-bp/blob/66627cb0f1747bf8a4deb6af146e3a9a5d47a3fe/reports/online_learning_2026_10_20.md).

Continuator still pins `5e8526473af13ee1106d616410284fa01f2c6d54` and calls
`from_sequences` after each learning invalidation. Upgrading the dependency alone
does not activate snapshot reuse. The deployed frontend pins Continuator
`dfeffa81a3aede834830d4b1c2c17ecbf39b7743`, which uses that older backend pin.

### Continuator prototype measurements

A temporary append-only integration probe used the actual 549-note MIDI fixture,
order four, eight successive 16-note additions and five isolated workers per
method. Added phrases included familiar contexts and transposed pitches.
Methods alternated between workers. Values are medians of worker medians and
include phrase storage, cache invalidation, model construction/update, graph
preparation and the first sample. MIDI parsing, realization, network and phrase-end
silence are excluded. These results describe the prototype, not deployed behavior.

| Request after learning | Full rebuild | Count and graph reuse | Latency reduction |
| --- | ---: | ---: | ---: |
| 64-note continuation with constrained final note | 29.6 ms | 22.7 ms | 23% |
| Until END, window 4–64 | 29.9 ms | 23.0 ms | 23% |
| 64-note continuation with a forbidden motif and final constraint | 47.3 ms | 40.7 ms | 14% |
| Unprompted 64-note generation with final constraint | 46.5 ms | 45.2 ms | 3% |

Counts-only updates saved about 1.7–1.8 ms for continuation and until-END;
ordinary graph reuse supplied most of the additional gain. Repeated cached
continuation remained about 0.54 ms. Regular constraints bypass Continuator's
inference cache, so repeated regular requests still prepare fresh inference.
Unprompted generation builds a separate model for its empirical initial
symbol distribution; retaining the ordinary model does not reuse that model's
graphs. Initially preserve this path and make no substantial unprompted or virtual
speedup claim.

[Raw measurements and methodology](../benchmarks/results/online_snapshot_integration_probe.json)
include all five worker medians. The prototype matched fresh model counts,
alphabet and forbidden markers after every addition; all methods produced the
same seeded outputs and released previous model objects. Compatibility checks
against the new backend passed 198 upstream tests, 84 Continuator tests and
55 frontend backend tests. Both pinned and new backend revisions passed the full
800-request audit, eight contract probes and 1,600 rolling notes, retaining the
same seeded acceptance outputs. These checks do not replace integration tests
for the eventual production adapter.

### Implementation plan

1. Update the immutable backend pin after validating the selected upstream
   revision. Keep all count, graph and BP algorithms in the dependency.
2. In `SequenceModel` and the MIDI facade, retain an already built symbolic
   snapshot for append-only learning. Queue encoded new phrases, coalesce
   successive additions, and apply `with_sequences` lazily on next generation.
   Initial folder/batch learning should still build once on first use.
3. Clear prepared plans, until backends, free-initial models, traces and
   diagnostics after learning. Keep only the current reusable source snapshot
   and pending additions; do not retain old BP tables or a chain of models.
4. Use full reconstruction for deletion, forgetting, memory reset and changes
   to order, viewpoint encoding or transform configuration. Preserve START/END
   padding, duplicate-phrase weights, new viewpoint handling, seeded generation,
   realization addresses and strict constraint behavior.
5. Add regressions for repeated and deferred additions, new symbols, cache
   invalidation, both generic and MIDI clients, forgetting/deletion, explicit
   and virtual augmentation, and release of obsolete snapshots. Compare counts,
   graph behavior and seeded samples with an independent full rebuild.
6. Run the baseline and required full acceptance/performance workload below.
   Record learn/update, graph preparation, BP preparation, sampling and retained
   memory separately. Update architecture/API documentation when integration is
   implemented, then update the frontend pin, run its integration tests and deploy.

## Remaining work

- Complete the required review of [Continuator PR #3](https://github.com/fpachet/continuator/pull/3).
  Backend PR #1 is merged, and the frontend already deploys the tested Continuator
  commit. The new snapshot integration described above remains a separate task.
- Test actual MIDI devices and streaming latency. The mocked output test covers
  metadata filtering, not hardware timing or device behavior.
- Implement the online snapshot integration above. Larger corpora, virtual
  transposition and regular-product constraints still have substantial first-request
  costs; only pursue further algorithms after profiling the intended workload.
- The default eight-entry, estimated 64 MiB table cache is not a process-memory
  limit. Graphs, candidate caches and native allocations consume additional memory.
  Until generation is bounded; streaming repeatedly requests finite continuations.

Reproduce:

```bash
python -m pytest -q
python -m benchmarks.engine_audit --samples 100 --repeats 5 --long-until \
  --output /tmp/continuator-audit.json
python -m build
python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz
```
