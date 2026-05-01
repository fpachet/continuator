# Agent Orientation

Before making architecture-level changes, read:

- `docs/current_architecture.md`
- `docs/class_map.md`

The current implementation should be treated as the "classic" Continuator
engine. Preserve the existing public imports unless a migration is explicitly
planned:

```python
from ctor.continuator import Continuator2
from ctor.variable_order_markov import Variable_order_Markov
```

Use the current test suite as the baseline:

```bash
python -m pytest -q
```
