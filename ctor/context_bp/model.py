from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
import random
from typing import Any, Callable, Hashable, Iterable, Mapping

from ctor.constraints import ConstraintProblem
from ctor.context_bp.context_graph import ContextCounts, ContextGraph
from ctor.context_bp.inference import ContextBPResult, NoFeasibleSequenceError, backward_messages, forward_backward
from ctor.context_bp.order_policy import CandidateSet, CandidateChoice, LongestFeasiblePolicy, OrderPolicy
from ctor.context_bp.vocabulary import Vocabulary


@dataclass(frozen=True)
class SampleStep:
    position: int
    symbol: Any
    order: int
    effective_order: int
    context: tuple[Any, ...]
    policy: str = ""
    candidate_orders: tuple[int, ...] = ()
    candidate_counts: tuple[int, ...] = ()
    skipped_orders: tuple[int, ...] = ()
    skipped_symbol: Any | None = None
    accepted_singleton: bool = False
    suppressed_skipped_symbol: bool = False


class ContextBPModel:
    """
    Generic variable-order Markov model with exact context-state BP.

    Generation starts from a hidden start context. Returned sequences contain
    only emitted symbols, never the hidden start symbol.
    """

    def __init__(
        self,
        kmax: int = 5,
        viewpoint_fn: Callable[[Any], Hashable] | None = None,
        seed: int | None = None,
        order_policy: OrderPolicy | None = None,
    ):
        if kmax < 1:
            raise ValueError("kmax must be at least 1")
        self.kmax = int(kmax)
        self.viewpoint_fn = viewpoint_fn
        self.order_policy = order_policy or LongestFeasiblePolicy()
        self.vocabulary = Vocabulary()
        self.counts = ContextCounts(self.kmax)
        self.symbol_counts: Counter[int] = Counter()
        self.rng = random.Random(seed)
        self.input_sequences: list[list[Any]] = []
        self.last_sample_trace: list[SampleStep] = []

    @property
    def start_symbol(self):
        return self.vocabulary.start_symbol

    @property
    def end_symbol(self):
        return self.vocabulary.end_symbol

    def get_viewpoint(self, obj):
        return obj if self.viewpoint_fn is None else self.viewpoint_fn(obj)

    def _encode_value(self, value: Any) -> int:
        if value is self.vocabulary.start_symbol:
            return self.vocabulary.start_id
        if value is self.vocabulary.end_symbol:
            return self.vocabulary.end_id
        try:
            return self.vocabulary.encode(value)
        except KeyError:
            return self.vocabulary.encode(self.get_viewpoint(value))

    def learn_sequence(self, sequence: Iterable[Any]) -> None:
        material = list(sequence)
        self.input_sequences.append(material)
        encoded = [self.vocabulary.start_id]
        encoded_material = [self.vocabulary.encode_or_add(self.get_viewpoint(item)) for item in material]
        encoded.extend(encoded_material)
        encoded.append(self.vocabulary.end_id)
        self.symbol_counts.update(encoded_material)
        self.counts.update_sequence(encoded)
        self.counts.update_sequence([self.vocabulary.end_id, self.vocabulary.end_id])

    def _effective_order(self, order: int | None = None) -> int:
        if order is None:
            return self.kmax
        if order < 1 or order > self.kmax:
            raise ValueError(f"order must be between 1 and {self.kmax}")
        return int(order)

    def _orders_to_try(self, order: int | None = None) -> range:
        effective_order = self._effective_order(order)
        return range(effective_order, 0, -1)

    def initial_context(
        self,
        prefix: Iterable[Any] | None = None,
        *,
        order: int | None = None,
    ) -> tuple[int, ...]:
        effective_order = self._effective_order(order)
        if prefix is None:
            prefix_items = []
        else:
            prefix_items = list(prefix)
        context = [self.vocabulary.start_id]
        context.extend(self._encode_value(item) for item in prefix_items)
        return tuple(context[-effective_order:])

    def compile_graph(
        self,
        *,
        prefix: Iterable[Any] | None = None,
        order: int | None = None,
    ) -> ContextGraph:
        effective_order = self._effective_order(order)
        return ContextGraph.from_counts(
            self.counts,
            initial_contexts=[self.initial_context(prefix, order=effective_order)],
            max_order=effective_order,
        )

    def infer(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None = None,
        constraints: ConstraintProblem | Mapping[int, Any] | None = None,
        order: int | None = None,
    ) -> tuple[ContextGraph, int, ContextBPResult]:
        allowed = self._allowed_symbols_by_position(length, constraints)
        last_error = None
        for effective_order in self._orders_to_try(order):
            graph = self.compile_graph(prefix=prefix, order=effective_order)
            initial_context = self.initial_context(prefix, order=effective_order)
            initial_state = graph.state_id(initial_context)
            try:
                result = forward_backward(
                    graph,
                    initial_state=initial_state,
                    length=length,
                    allowed_symbols_by_position=allowed,
                    vocab_size=len(self.vocabulary),
                )
                return graph, initial_state, result
            except NoFeasibleSequenceError as e:
                last_error = e
        raise NoFeasibleSequenceError("No context path satisfies the constraints at any order.") from last_error

    def sample_sequence(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None = None,
        constraints: ConstraintProblem | Mapping[int, Any] | None = None,
        initial_mode: str = "start",
        raise_on_fail: bool = False,
    ) -> list[Any] | None:
        try:
            if initial_mode == "start":
                decoded = self._sample_with_stepwise_order_backoff(
                    length,
                    prefix=prefix,
                    constraints=constraints,
                    raise_on_fail=raise_on_fail,
                )
            elif initial_mode == "free":
                decoded = self._sample_with_free_initial(
                    length,
                    prefix=prefix,
                    constraints=constraints,
                    raise_on_fail=raise_on_fail,
                )
            else:
                raise ValueError("initial_mode must be 'start' or 'free'")
        except NoFeasibleSequenceError:
            if raise_on_fail:
                raise
            return None

        if decoded is None:
            return None
        if not self.sequence_satisfies_constraints(decoded, constraints):
            if raise_on_fail:
                raise NoFeasibleSequenceError("Sampled sequence violates constraints.")
            return None
        return decoded

    def sample_sequence_with_trace(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None = None,
        constraints: ConstraintProblem | Mapping[int, Any] | None = None,
        initial_mode: str = "start",
        raise_on_fail: bool = False,
    ) -> tuple[list[Any], list[SampleStep]] | None:
        sequence = self.sample_sequence(
            length,
            prefix=prefix,
            constraints=constraints,
            initial_mode=initial_mode,
            raise_on_fail=raise_on_fail,
        )
        if sequence is None:
            return None
        return sequence, list(self.last_sample_trace)

    def last_sample_trace_as_dicts(self) -> list[dict[str, Any]]:
        """Return the last generation trace in a JSON-friendly shape."""
        return [self._sample_step_as_dict(step) for step in self.last_sample_trace]

    def continue_until_end(
        self,
        prefix: Iterable[Any] | None = None,
        *,
        min_length: int = 1,
        max_length: int = 64,
        end_symbol: Any | None = None,
        raise_on_fail: bool = False,
    ) -> list[Any] | None:
        """
        Generate until the first hit of `end_symbol` inside a length window.

        The prefix is conditioning context only and is not included in the
        returned sequence. The returned sequence includes the end symbol.
        """
        result = self.continue_until_end_with_trace(
            prefix=prefix,
            min_length=min_length,
            max_length=max_length,
            end_symbol=end_symbol,
            raise_on_fail=raise_on_fail,
        )
        if result is None:
            return None
        sequence, _ = result
        return sequence

    def continue_until_end_with_trace(
        self,
        prefix: Iterable[Any] | None = None,
        *,
        min_length: int = 1,
        max_length: int = 64,
        end_symbol: Any | None = None,
        raise_on_fail: bool = False,
    ) -> tuple[list[Any], list[SampleStep]] | None:
        """
        Generate until first target hit and return the per-step policy trace.

        Lengths are selected from first-hit BP path masses. Once a length is
        selected, the actual sequence is sampled with the same stepwise order
        policy used by fixed-length constrained generation.
        """
        _, _, weighted_lengths = self._first_hit_path_data(
            prefix=prefix,
            min_length=min_length,
            max_length=max_length,
            end_symbol=end_symbol,
            raise_on_fail=raise_on_fail,
        )
        if not weighted_lengths:
            return None

        remaining_lengths = dict(weighted_lengths)
        while remaining_lengths:
            lengths = list(remaining_lengths)
            weights = [remaining_lengths[length][0] for length in lengths]
            chosen_length = self.rng.choices(lengths, weights=weights, k=1)[0]
            _, allowed, _ = remaining_lengths[chosen_length]
            sequence = self._sample_with_allowed_stepwise(
                chosen_length,
                prefix=prefix,
                allowed_symbols_by_position=allowed,
                raise_on_fail=False,
            )
            if sequence is not None:
                return sequence, list(self.last_sample_trace)
            del remaining_lengths[chosen_length]

        if raise_on_fail:
            raise NoFeasibleSequenceError("No first-hit path satisfies the requested window at any order.")
        return None

    def first_hit_lengths(
        self,
        prefix: Iterable[Any] | None = None,
        *,
        min_length: int = 1,
        max_length: int = 64,
        end_symbol: Any | None = None,
    ) -> list[int]:
        """Return feasible first-hit lengths for the given prefix and target."""
        _, _, weighted_lengths = self._first_hit_path_data(
            prefix=prefix,
            min_length=min_length,
            max_length=max_length,
            end_symbol=end_symbol,
            raise_on_fail=False,
        )
        return list(weighted_lengths)

    def _sample_with_stepwise_order_backoff(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None,
        constraints: ConstraintProblem | Mapping[int, Any] | None,
        raise_on_fail: bool,
    ) -> list[Any] | None:
        allowed = self._allowed_symbols_by_position(length, constraints)
        return self._sample_with_allowed_stepwise(
            length,
            prefix=prefix,
            allowed_symbols_by_position=allowed,
            raise_on_fail=raise_on_fail,
        )

    def _sample_with_free_initial(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None,
        constraints: ConstraintProblem | Mapping[int, Any] | None,
        raise_on_fail: bool,
    ) -> list[Any] | None:
        if prefix is not None:
            return self._sample_with_stepwise_order_backoff(
                length,
                prefix=prefix,
                constraints=constraints,
                raise_on_fail=raise_on_fail,
            )
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            self.last_sample_trace = []
            return []

        allowed = self._allowed_symbols_by_position(length, constraints)
        candidate_ids = [
            symbol_id
            for symbol_id in allowed[0]
            if symbol_id != self.vocabulary.start_id
        ]
        weighted_candidates = []
        for symbol_id in candidate_ids:
            weight = self._free_initial_candidate_weight(symbol_id, allowed[1:])
            if weight > 0:
                weighted_candidates.append((symbol_id, weight))

        while weighted_candidates:
            ids, weights = zip(*weighted_candidates)
            first_symbol_id = self.rng.choices(ids, weights=weights, k=1)[0]
            first_symbol = self.vocabulary.decode(first_symbol_id)
            suffix = self._sample_with_allowed_stepwise(
                length - 1,
                prefix=[first_symbol],
                allowed_symbols_by_position=allowed[1:],
                raise_on_fail=False,
            )
            if suffix is not None:
                first_step = SampleStep(
                    position=0,
                    symbol=first_symbol,
                    order=0,
                    effective_order=0,
                    context=(),
                    policy="free_initial",
                    candidate_orders=(0,),
                    candidate_counts=(len(weighted_candidates),),
                )
                suffix_trace = [replace(step, position=step.position + 1) for step in self.last_sample_trace]
                self.last_sample_trace = [first_step] + suffix_trace
                return [first_symbol] + suffix
            weighted_candidates = [
                candidate
                for candidate in weighted_candidates
                if candidate[0] != first_symbol_id
            ]

        self.last_sample_trace = []
        if raise_on_fail:
            raise NoFeasibleSequenceError("No context path satisfies the constraints from any initial symbol.")
        return None

    def _free_initial_candidate_weight(self, first_symbol_id: int, suffix_allowed: list[set[int]]) -> float:
        if not suffix_allowed:
            return float(self.symbol_counts.get(first_symbol_id, 1))

        prefix = [self.vocabulary.decode(first_symbol_id)]
        last_error = None
        for order in self._orders_to_try():
            graph = self.compile_graph(prefix=prefix, order=order)
            initial_context = self.initial_context(prefix, order=order)
            initial_state = graph.state_id(initial_context)
            try:
                result = forward_backward(
                    graph,
                    initial_state=initial_state,
                    length=len(suffix_allowed),
                    allowed_symbols_by_position=suffix_allowed,
                    vocab_size=len(self.vocabulary),
                )
                return float(self.symbol_counts.get(first_symbol_id, 1)) * result.path_mass
            except NoFeasibleSequenceError as e:
                last_error = e
        return 0.0

    def _sample_with_allowed_stepwise(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None,
        allowed_symbols_by_position: list[set[int]],
        raise_on_fail: bool,
    ) -> list[Any] | None:
        if len(allowed_symbols_by_position) != length:
            raise ValueError("allowed_symbols_by_position must match length")
        order_data = {}
        for order in self._orders_to_try():
            graph = self.compile_graph(prefix=prefix, order=order)
            order_data[order] = (
                graph,
                backward_messages(
                    graph,
                    length=length,
                    allowed_symbols_by_position=allowed_symbols_by_position,
                ),
            )

        history = [self.vocabulary.start_id]
        if prefix is not None:
            history.extend(self._encode_value(item) for item in prefix)

        sequence: list[int] = []
        trace: list[SampleStep] = []
        for position in range(length):
            candidate_sets = self._candidate_sets_for_position(
                order_data,
                history,
                allowed_symbols_by_position,
                position,
            )

            chosen = self.order_policy.choose(candidate_sets, self.rng)
            if chosen is None:
                self.last_sample_trace = trace
                if raise_on_fail:
                    raise NoFeasibleSequenceError("No context path satisfies the constraints at any order.")
                return None

            candidate_set = chosen.candidate_set
            edge = chosen.edge
            sequence.append(edge.symbol)
            trace.append(self._make_sample_step(position, chosen))
            history.append(edge.symbol)

        self.last_sample_trace = trace
        return [self.vocabulary.decode(symbol) for symbol in sequence]

    def _candidate_sets_for_position(
        self,
        order_data: dict[int, tuple[ContextGraph, Any]],
        history: list[int],
        allowed_symbols_by_position: list[set[int]],
        position: int,
    ) -> list[CandidateSet]:
        candidate_sets = []
        max_context_order = min(self.kmax, len(history))
        for order in range(max_context_order, 0, -1):
            graph, backward = order_data[order]
            context = tuple(history[-order:])
            try:
                state = graph.state_id(context)
            except KeyError:
                continue

            candidates = []
            weights = []
            for edge in graph.outgoing[state]:
                if edge.symbol not in allowed_symbols_by_position[position]:
                    continue
                weight = edge.weight * backward[position + 1, edge.dst]
                if weight <= 0:
                    continue
                candidates.append(edge)
                weights.append(weight)
            if candidates:
                candidate_sets.append(
                    CandidateSet(
                        order=order,
                        graph=graph,
                        state=state,
                        edges=tuple(candidates),
                        weights=tuple(weights),
                    )
                )
        return candidate_sets

    def _make_sample_step(self, position: int, chosen: CandidateChoice) -> SampleStep:
        candidate_set = chosen.candidate_set
        graph = candidate_set.graph
        state = candidate_set.state
        edge = chosen.edge
        decision = chosen.decision
        skipped_symbol = (
            None
            if decision.skipped_symbol is None
            else self.vocabulary.decode(decision.skipped_symbol)
        )
        return SampleStep(
            position=position,
            symbol=self.vocabulary.decode(edge.symbol),
            order=edge.order,
            effective_order=graph.kmax,
            context=tuple(self.vocabulary.decode(symbol) for symbol in graph.contexts[state]),
            policy=decision.policy_name,
            candidate_orders=decision.candidate_orders,
            candidate_counts=decision.candidate_counts,
            skipped_orders=decision.skipped_orders,
            skipped_symbol=skipped_symbol,
            accepted_singleton=decision.accepted_singleton,
            suppressed_skipped_symbol=decision.suppressed_skipped_symbol,
        )

    def _sample_step_as_dict(self, step: SampleStep) -> dict[str, Any]:
        return {
            "position": step.position,
            "symbol": self._trace_value(step.symbol),
            "order": step.order,
            "effective_order": step.effective_order,
            "context": [self._trace_value(symbol) for symbol in step.context],
            "policy": step.policy,
            "candidate_orders": list(step.candidate_orders),
            "candidate_counts": list(step.candidate_counts),
            "skipped_orders": list(step.skipped_orders),
            "skipped_symbol": self._trace_value(step.skipped_symbol),
            "accepted_singleton": step.accepted_singleton,
            "suppressed_skipped_symbol": step.suppressed_skipped_symbol,
        }

    @staticmethod
    def _trace_value(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "item"):
            try:
                scalar_value = value.item()
            except ValueError:
                scalar_value = None
            if isinstance(scalar_value, (bool, int, float, str)):
                return scalar_value
        if isinstance(value, tuple):
            return [ContextBPModel._trace_value(item) for item in value]
        return repr(value)

    def _first_hit_path_data(
        self,
        *,
        prefix: Iterable[Any] | None,
        min_length: int,
        max_length: int,
        end_symbol: Any | None,
        raise_on_fail: bool,
    ) -> tuple[ContextGraph, int, dict[int, tuple[float, list[set[int]], ContextBPResult]]]:
        if min_length < 1:
            raise ValueError("min_length must be at least 1")
        if max_length < min_length:
            raise ValueError("max_length must be greater than or equal to min_length")

        try:
            target_id = self.vocabulary.end_id if end_symbol is None else self._encode_value(end_symbol)
        except KeyError as e:
            if raise_on_fail:
                raise NoFeasibleSequenceError("Unknown end symbol.") from e
            graph = self.compile_graph(prefix=prefix)
            return graph, graph.state_id(self.initial_context(prefix, order=graph.kmax)), {}
        if target_id == self.vocabulary.start_id:
            raise ValueError("end_symbol cannot be the hidden start symbol")

        last_graph = None
        last_initial_state = None
        for effective_order in self._orders_to_try():
            graph = self.compile_graph(prefix=prefix, order=effective_order)
            initial_state = graph.state_id(self.initial_context(prefix, order=effective_order))
            last_graph = graph
            last_initial_state = initial_state
            reachable = graph.first_hit_reachable_to_symbol(target_id, max_length)
            if not graph.can_reach_between(reachable, initial_state, min_length, max_length):
                continue

            weighted_lengths: dict[int, tuple[float, list[set[int]], ContextBPResult]] = {}
            for length in range(min_length, max_length + 1):
                if initial_state not in reachable[length]:
                    continue
                allowed = self._first_hit_allowed_symbols_by_position(length, target_id)
                try:
                    inference = forward_backward(
                        graph,
                        initial_state=initial_state,
                        length=length,
                        allowed_symbols_by_position=allowed,
                        vocab_size=len(self.vocabulary),
                    )
                except NoFeasibleSequenceError:
                    continue
                if inference.path_mass > 0:
                    weighted_lengths[length] = (inference.path_mass, allowed, inference)

            if weighted_lengths:
                return graph, initial_state, weighted_lengths

        if raise_on_fail:
            raise NoFeasibleSequenceError("No first-hit path satisfies the requested window at any order.")
        if last_graph is None or last_initial_state is None:
            last_graph = self.compile_graph(prefix=prefix)
            last_initial_state = last_graph.state_id(self.initial_context(prefix, order=last_graph.kmax))
        return last_graph, last_initial_state, {}

    def _first_hit_allowed_symbols_by_position(
        self,
        length: int,
        target_id: int,
    ) -> list[set[int]]:
        if length < 1:
            raise ValueError("length must be at least 1")
        allowed = self._allowed_symbols_by_position(length, constraints=None)
        for position in range(length - 1):
            allowed[position].discard(target_id)
        allowed[length - 1] = {target_id}
        return allowed

    def symbol_marginals(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None = None,
        constraints: ConstraintProblem | Mapping[int, Any] | None = None,
    ) -> list[dict[Any, float]]:
        _, _, inference = self.infer(length, prefix=prefix, constraints=constraints)
        marginals: list[dict[Any, float]] = []
        for row in inference.symbol_marginals:
            distribution = {}
            for symbol_id, probability in enumerate(row):
                if probability > 0:
                    distribution[self.vocabulary.decode(symbol_id)] = float(probability)
            marginals.append(distribution)
        return marginals

    def _allowed_symbols_by_position(
        self,
        length: int,
        constraints: ConstraintProblem | Mapping[int, Any] | None,
    ) -> list[set[int]]:
        if length < 0:
            raise ValueError("length must be non-negative")
        allowed = [
            set(range(len(self.vocabulary)))
            - {self.vocabulary.start_id, self.vocabulary.end_id}
            for _ in range(length)
        ]
        if constraints is None:
            return allowed

        if isinstance(constraints, ConstraintProblem):
            if constraints.length is not None and constraints.length != length:
                raise ValueError(
                    f"ConstraintProblem length {constraints.length} does not match requested length {length}"
                )
            items = constraints.allowed_values_by_position.items()
        else:
            items = ((position, {value}) for position, value in constraints.items())

        for position, values in items:
            if position < 0 or position >= length:
                raise IndexError(f"constraint position {position} is outside length {length}")
            encoded_values = {self._encode_value(value) for value in values}
            if not encoded_values:
                raise ValueError(f"position {position} has an empty allowed set")
            allowed[position] = encoded_values
        return allowed

    @staticmethod
    def sequence_satisfies_constraints(
        sequence: list[Any],
        constraints: ConstraintProblem | Mapping[int, Any] | None = None,
    ) -> bool:
        if constraints is None:
            return True
        if isinstance(constraints, ConstraintProblem):
            items = constraints.allowed_values_by_position.items()
        else:
            items = ((position, {value}) for position, value in constraints.items())
        for position, values in items:
            if position < 0 or position >= len(sequence):
                return False
            if sequence[position] not in values:
                return False
        return True
