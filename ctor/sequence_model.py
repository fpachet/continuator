"""The single symbolic generation adapter used by MIDI and generic clients."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping
from dataclasses import astuple, dataclass, is_dataclass, replace
import random

from vo_regular_bp import (
    ConstraintSet, OrderStackModel, SingletonAvoidingBackoffPolicy,
    prepare_constrained_order_stack_plan, prepare_until_order_stack,
)

from ctor.belief_propag import NoSolutionErrorInBP
from ctor.constraints import ConstraintProblem


@dataclass(frozen=True)
class Boundary:
    name: str

    def __repr__(self):
        return f"<{self.name}>"


class SequenceModel:
    """Learn hashable symbols and generate through the external VO order stack.

    Unprompted fixed sampling has free initial symbols. A prefix (including an
    explicit START) conditions generation without being returned. Until-mode
    without a prefix starts at a learned phrase boundary. Hard constraints are
    never silently relaxed. Learning invalidates inference immediately and
    defers rebuilding the symbolic model until it is actually needed.
    """

    def __init__(self, *, kmax=4, viewpoint_fn=None, seed=0, policy=None,
                 cache_size=8, cache_bytes=64 * 1024 * 1024,
                 start_symbol=None, end_symbol=None, _model_factory=None):
        if not isinstance(kmax, int) or kmax < 1:
            raise ValueError("kmax must be a positive integer")
        if cache_size < 0 or cache_bytes < 0:
            raise ValueError("cache limits must be non-negative")
        self.kmax = kmax
        self.viewpoint_fn = viewpoint_fn
        self.start_symbol = Boundary("START") if start_symbol is None else start_symbol
        self.end_symbol = Boundary("END") if end_symbol is None else end_symbol
        self._free_symbol = Boundary("FREE")
        self.rng = random.Random(seed)
        self.policy = policy or SingletonAvoidingBackoffPolicy()
        self.cache_size, self.cache_bytes = cache_size, cache_bytes
        self._model_factory = _model_factory
        self._sequences = []
        self._symbols = set()
        self._cache = OrderedDict()
        self._cache_weight = 0
        self._model = self._free_model = None
        self.revision = 0
        self.cache_hits = self.cache_misses = 0
        self._reset_generation()

    @property
    def start_padding(self):
        return self.start_symbol

    @property
    def end_padding(self):
        return self.end_symbol

    @property
    def order_model(self):
        if self._model is None:
            self._model = self._model_factory() if self._model_factory else OrderStackModel.from_sequences(
                self._sequences, max_order=self.kmax,
                start_symbol=self.start_symbol, end_symbol=self.end_symbol,
            )
        return self._model

    def invalidate(self):
        self.revision += 1
        self._model = self._free_model = None
        self._cache.clear()
        self._cache_weight = 0
        self._reset_generation()

    def learn_sequence(self, sequence):
        material = tuple(self.get_viewpoint(item) for item in sequence)
        if material:
            self._sequences.append(material)
            self._symbols.update(material)
            self.invalidate()

    def learn_sequences(self, sequences):
        for sequence in sequences:
            self.learn_sequence(sequence)

    def clear_memory(self):
        self._sequences.clear()
        self._symbols.clear()
        self.invalidate()

    def get_viewpoint(self, item):
        if item is self.start_symbol or item is self.end_symbol:
            return item
        return self.viewpoint_fn(item) if self.viewpoint_fn else item

    def _prefix_viewpoint(self, item):
        # Learned viewpoints take precedence when original objects and
        # viewpoints share a type. Learning always maps original objects.
        try:
            if item in self._symbols:
                return item
        except TypeError:
            pass
        return self.get_viewpoint(item)

    def has_viewpoint(self, symbol):
        return symbol in self.order_model.alphabet or symbol is self.start_symbol

    def _free_initial_model(self):
        if self._free_model is None:
            model = self.order_model
            # Add a hidden root with empirical symbol priors. All inference,
            # regular constraints and order selection remain in vo_regular_bp.
            counts_by_context = getattr(model, "counts", None)
            if counts_by_context is None:
                counts_by_context = {c: model.augmented_counts(c) for c in model.iter_contexts(self.kmax)}
            priors = Counter()
            for context, counts in counts_by_context.items():
                if len(context) == 1:
                    priors.update({s: n for s, n in counts.items()
                                   if s not in (self.start_symbol, self.end_symbol)})
            priors[self.start_symbol] = sum(counts_by_context.get((self.start_symbol,), {}).values())
            self._free_model = OrderStackModel(
                {**counts_by_context, (self._free_symbol,): priors},
                max_order=self.kmax, forbidden_symbols=model.forbidden_symbols | {self._free_symbol},
            )
        return self._free_model

    def _constraints(self, constraints, length=None):
        if isinstance(constraints, ConstraintProblem):
            if length is not None and constraints.length not in (None, length):
                raise ValueError("ConstraintProblem.length must match generation length")
            return ConstraintSet(positional={p: frozenset(v) for p, v in constraints.allowed_values_by_position.items()})
        if isinstance(constraints, ConstraintSet):
            return replace(constraints,
                           positional={p: mask if callable(mask) else frozenset(mask)
                                       for p, mask in constraints.positional.items()},
                           forbidden_substrings=tuple(tuple(p) for p in constraints.forbidden_substrings))
        if constraints is None:
            return ConstraintSet()
        if isinstance(constraints, Mapping):
            return ConstraintSet(positional={p: frozenset((v,)) for p, v in constraints.items()})
        raise TypeError("constraints must be a mapping, ConstraintProblem or ConstraintSet")

    def _fingerprint(self, constraints):
        # Callable masks/automata can depend on mutable state. Do not cache them.
        if (constraints.regular_acceptors or constraints.forbidden_substrings
                or constraints.meter or constraints.cumulative_meter):
            return None
        if any(callable(v) for v in constraints.positional.values()):
            return None
        try:
            policy = (type(self.policy), astuple(self.policy)) if is_dataclass(self.policy) else None
            if policy is None:
                return None
            key = (tuple(sorted((p, frozenset(v)) for p, v in constraints.positional.items())),
                   tuple(tuple(p) for p in constraints.forbidden_substrings), policy)
            hash(key)
            return key
        except TypeError:
            return None

    def _prepare(self, key, build, *, length):
        if key is not None and key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key][0]
        self.cache_misses += 1
        value = build()
        if key is None or not self.cache_size:
            return value
        # One Python float plus list slot is approximately 32 bytes. Count
        # another table only if log-space messages were actually needed.
        # Until backends share tables; inspect one longest-horizon result.
        lower = value.plan if hasattr(value, "plan") else value.backends[-1].result
        weight = sum((length + 1) * len(rows[0]) * 32
                     * (2 if getattr(rows, "log_values", None) is not None else 1)
                     for rows in lower.backwards.values())
        if key is not None and self.cache_size and weight <= self.cache_bytes:
            while self._cache and (len(self._cache) >= self.cache_size or self._cache_weight + weight > self.cache_bytes):
                _, (_, removed_weight) = self._cache.popitem(last=False)
                self._cache_weight -= removed_weight
            self._cache[key] = (value, weight)
            self._cache_weight += weight
        return value

    def sample_sequence(self, length, *, prefix=None, constraints=None, start_vp=None,
                        raise_on_fail=False):
        if not isinstance(length, int) or length < 0:
            raise ValueError("length must be a non-negative integer")
        self._reset_generation()
        spec = self._constraints(constraints, length)
        if any(not isinstance(p, int) or not 0 <= p < length for p in spec.positional):
            raise ValueError("constraint position outside generated sequence")
        if length == 0:
            return []
        context = tuple(self._prefix_viewpoint(v) for v in ([start_vp] if start_vp is not None else prefix or ()))
        free = not context
        model = self._free_initial_model() if free else self.order_model
        if free:
            context = (self._free_symbol,)
        allowed = {}
        for position, symbol in ((0, self.start_symbol), (length - 1, self.end_symbol)):
            mask = spec.positional.get(position)
            if mask is not None and (bool(mask(symbol)) if callable(mask) else symbol in mask):
                allowed.setdefault(position, set()).add(symbol)
        fingerprint = self._fingerprint(spec)
        key = None if fingerprint is None else ("fixed", free, length, fingerprint)
        try:
            plan = self._prepare(key, lambda: prepare_constrained_order_stack_plan(
                model, spec, length=length, policy=self.policy, allowed_forbidden_symbols=allowed,
            ), length=length)
            return self._sample(plan.for_prefix(context))
        except ValueError as exc:
            return self._no_solution(exc, raise_on_fail)

    def continue_sequence(self, prefix, length=50, constraints=None, **kwargs):
        return self.sample_sequence(length, prefix=prefix, constraints=constraints, **kwargs)

    def continue_until(self, prefix=None, *, stop, min_length=1, max_length=64,
                       constraints=None, raise_on_fail=False):
        if not isinstance(min_length, int) or not isinstance(max_length, int) or not 1 <= min_length <= max_length:
            raise ValueError("require 1 <= min_length <= max_length")
        self._reset_generation()
        spec = None if constraints is None else self._constraints(constraints)
        context = tuple(self._prefix_viewpoint(v) for v in prefix or (self.start_symbol,))
        model = self.order_model
        if not callable(stop):
            try:
                known = stop in model.alphabet
            except TypeError:
                known = False
            if not known and not isinstance(stop, (str, bytes)):
                try:
                    stop = frozenset(stop)
                except TypeError:
                    pass
        fingerprint = self._fingerprint(spec or ConstraintSet())
        key = None if callable(stop) or fingerprint is None else ("until", context, stop, min_length, max_length, fingerprint)
        # Until with regular/positional conditions allocates one backend per
        # horizon. Keep it uncached rather than underestimating its memory.
        if spec is not None:
            key = None
        try:
            backend = self._prepare(key, lambda: prepare_until_order_stack(
                model, prefix=context, stop=stop, min_length=min_length, max_length=max_length,
                constraints=spec, policy=self.policy,
            ), length=max_length)
            return self._sample(backend)
        except ValueError as exc:
            return self._no_solution(exc, raise_on_fail)

    def continue_until_end(self, prefix=None, min_length=1, max_length=64, end_vp=None, *, raise_on_fail=False):
        return self.continue_until(prefix, stop=self.end_symbol if end_vp is None else end_vp,
                                   min_length=min_length, max_length=max_length, raise_on_fail=raise_on_fail)

    def _sample(self, backend):
        sequence, self.last_trace = backend.sample_with_trace(rng=self.rng)
        self._last_backend = backend
        return list(sequence)

    def _reset_generation(self):
        self.last_trace = ()
        self._last_backend = None
        self._diagnostics = None

    def _no_solution(self, exc, raise_on_fail):
        self._reset_generation()
        if raise_on_fail:
            raise NoSolutionErrorInBP("No sequence satisfies the constraints and prefix.") from exc
        return None

    def get_last_generation_diagnostics(self):
        if self._diagnostics is None:
            self._diagnostics = self._last_backend.diagnostics.as_dict() if self._last_backend else {}
        return {**self._diagnostics, "model_revision": self.revision, "cache_entries": len(self._cache),
                "cache_estimated_bytes": self._cache_weight, "cache_hits": self.cache_hits, "cache_misses": self.cache_misses}
