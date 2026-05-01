from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Hashable, Iterable, Mapping

from ctor.constraints import ConstraintProblem
from ctor.core.context_graph import ContextCounts, ContextGraph
from ctor.core.inference import ContextBPResult, NoFeasibleSequenceError, backward_messages, forward_backward
from ctor.core.vocabulary import Vocabulary


@dataclass(frozen=True)
class SampleStep:
    position: int
    symbol: Any
    order: int
    effective_order: int
    context: tuple[Any, ...]


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
    ):
        if kmax < 1:
            raise ValueError("kmax must be at least 1")
        self.kmax = int(kmax)
        self.viewpoint_fn = viewpoint_fn
        self.vocabulary = Vocabulary()
        self.counts = ContextCounts(self.kmax)
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
        encoded.extend(self.vocabulary.encode_or_add(self.get_viewpoint(item)) for item in material)
        encoded.append(self.vocabulary.end_id)
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
        raise_on_fail: bool = False,
    ) -> list[Any] | None:
        try:
            decoded = self._sample_with_stepwise_order_backoff(
                length,
                prefix=prefix,
                constraints=constraints,
                raise_on_fail=raise_on_fail,
            )
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
        raise_on_fail: bool = False,
    ) -> tuple[list[Any], list[SampleStep]] | None:
        sequence = self.sample_sequence(
            length,
            prefix=prefix,
            constraints=constraints,
            raise_on_fail=raise_on_fail,
        )
        if sequence is None:
            return None
        return sequence, list(self.last_sample_trace)

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
        graph, initial_state, weighted_lengths = self._first_hit_path_data(
            prefix=prefix,
            min_length=min_length,
            max_length=max_length,
            end_symbol=end_symbol,
            raise_on_fail=raise_on_fail,
        )
        if not weighted_lengths:
            return None

        lengths = list(weighted_lengths)
        weights = [weighted_lengths[length][0] for length in lengths]
        chosen_length = self.rng.choices(lengths, weights=weights, k=1)[0]
        _, allowed, inference = weighted_lengths[chosen_length]
        return self._sample_with_allowed(
            graph,
            initial_state,
            inference,
            allowed,
            raise_on_fail=raise_on_fail,
        )

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
        order_data = {}
        for order in self._orders_to_try():
            graph = self.compile_graph(prefix=prefix, order=order)
            order_data[order] = (
                graph,
                backward_messages(
                    graph,
                    length=length,
                    allowed_symbols_by_position=allowed,
                ),
            )

        history = [self.vocabulary.start_id]
        if prefix is not None:
            history.extend(self._encode_value(item) for item in prefix)

        sequence: list[int] = []
        trace: list[SampleStep] = []
        for position in range(length):
            chosen = None
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
                    if edge.symbol not in allowed[position]:
                        continue
                    weight = edge.weight * backward[position + 1, edge.dst]
                    if weight <= 0:
                        continue
                    candidates.append(edge)
                    weights.append(weight)
                if candidates:
                    edge = self.rng.choices(candidates, weights=weights, k=1)[0]
                    chosen = graph, state, edge
                    break

            if chosen is None:
                self.last_sample_trace = trace
                if raise_on_fail:
                    raise NoFeasibleSequenceError("No context path satisfies the constraints at any order.")
                return None

            graph, state, edge = chosen
            sequence.append(edge.symbol)
            trace.append(
                SampleStep(
                    position=position,
                    symbol=self.vocabulary.decode(edge.symbol),
                    order=edge.order,
                    effective_order=graph.kmax,
                    context=tuple(self.vocabulary.decode(symbol) for symbol in graph.contexts[state]),
                )
            )
            history.append(edge.symbol)

        self.last_sample_trace = trace
        return [self.vocabulary.decode(symbol) for symbol in sequence]

    def _sample_with_allowed(
        self,
        graph: ContextGraph,
        initial_state: int,
        inference: ContextBPResult,
        allowed: list[set[int]],
        *,
        raise_on_fail: bool,
    ) -> list[Any] | None:
        state = initial_state
        sequence: list[int] = []
        trace: list[SampleStep] = []
        for position in range(len(allowed)):
            candidates = []
            weights = []
            for edge in graph.outgoing[state]:
                if edge.symbol not in allowed[position]:
                    continue
                weight = edge.weight * inference.backward[position + 1, edge.dst]
                if weight <= 0:
                    continue
                candidates.append(edge)
                weights.append(weight)
            if not candidates:
                if raise_on_fail:
                    raise NoFeasibleSequenceError("No context path satisfies the constraints.")
                return None
            edge = self.rng.choices(candidates, weights=weights, k=1)[0]
            sequence.append(edge.symbol)
            trace.append(
                SampleStep(
                    position=position,
                    symbol=self.vocabulary.decode(edge.symbol),
                    order=edge.order,
                    effective_order=graph.kmax,
                    context=tuple(self.vocabulary.decode(symbol) for symbol in graph.contexts[state]),
                )
            )
            state = edge.dst

        self.last_sample_trace = trace
        return [self.vocabulary.decode(symbol) for symbol in sequence]

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
