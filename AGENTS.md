# Agent orientation

Before architecture-level changes, read:

- `docs/current_architecture.md`
- `docs/class_map.md`
- `docs/public_api.md`
- `docs/migration.md`

There is one maintained generation engine: external `vo_regular_bp`. Keep BP
and automaton algorithms in that dependency. Preserve the stable MIDI import:

```python
from ctor.continuator import Continuator2
```

Generic code uses `ctor.sequence_model.SequenceModel`.
`ctor.variable_order_markov.Variable_order_Markov` is a deprecated
generation-only adapter, not the historical engine. The classic and context-BP
implementations and their comparison audit live in the archive tag documented
in `docs/migration.md`.

Run the active baseline:

```bash
python -m pytest -q
```

Run the reproducible acceptance/performance workload when changing generation,
caches, memory or realization:

```bash
python -m benchmarks.engine_audit --samples 100 --repeats 5 --long-until --output /tmp/continuator-audit.json
```
