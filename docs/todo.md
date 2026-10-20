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

## Remaining work

- Review and merge the upstream and Continuator changes, then update the frontend's
  immutable Continuator pin and run its deployment integration checks.
- Test actual MIDI devices and streaming latency. The mocked output test covers
  metadata filtering, not hardware timing or device behavior.
- Large corpora, virtual transposition and regular-product constraints still have
  substantial first-request costs. Incremental model/graph updates are a possible
  separate upstream project; first profile the intended workload. Learning must
  continue to invalidate inference, so stale constraints/counts are never reused.
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
