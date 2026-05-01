from __future__ import annotations

import random
from typing import Any, Callable, Hashable, Iterable, Mapping

from ctor.constraints import ConstraintProblem
from ctor.core.context_graph import ContextCounts, ContextGraph
from ctor.core.inference import ContextBPResult, NoFeasibleSequenceError, forward_backward
from ctor.core.vocabulary import Vocabulary


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

    @property
    def start_symbol(self):
        return self.vocabulary.start_symbol

    @property
    def end_symbol(self):
        return self.vocabulary.end_symbol

    def get_viewpoint(self, obj):
        return obj if self.viewpoint_fn is None else self.viewpoint_fn(obj)

    def learn_sequence(self, sequence: Iterable[Any]) -> None:
        material = list(sequence)
        self.input_sequences.append(material)
        encoded = [self.vocabulary.start_id]
        encoded.extend(self.vocabulary.encode_or_add(self.get_viewpoint(item)) for item in material)
        encoded.append(self.vocabulary.end_id)
        self.counts.update_sequence(encoded)
        self.counts.update_sequence([self.vocabulary.end_id, self.vocabulary.end_id])

    def initial_context(self, prefix: Iterable[Any] | None = None) -> tuple[int, ...]:
        if prefix is None:
            prefix_items = []
        else:
            prefix_items = list(prefix)
        context = [self.vocabulary.start_id]
        context.extend(self.vocabulary.encode(self.get_viewpoint(item)) for item in prefix_items)
        return tuple(context[-self.kmax:])

    def compile_graph(self, *, prefix: Iterable[Any] | None = None) -> ContextGraph:
        return ContextGraph.from_counts(
            self.counts,
            initial_contexts=[self.initial_context(prefix)],
        )

    def infer(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None = None,
        constraints: ConstraintProblem | Mapping[int, Any] | None = None,
    ) -> tuple[ContextGraph, int, ContextBPResult]:
        graph = self.compile_graph(prefix=prefix)
        initial_context = self.initial_context(prefix)
        initial_state = graph.state_id(initial_context)
        allowed = self._allowed_symbols_by_position(length, constraints)
        result = forward_backward(
            graph,
            initial_state=initial_state,
            length=length,
            allowed_symbols_by_position=allowed,
            vocab_size=len(self.vocabulary),
        )
        return graph, initial_state, result

    def sample_sequence(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None = None,
        constraints: ConstraintProblem | Mapping[int, Any] | None = None,
        raise_on_fail: bool = False,
    ) -> list[Any] | None:
        try:
            graph, initial_state, inference = self.infer(
                length,
                prefix=prefix,
                constraints=constraints,
            )
        except NoFeasibleSequenceError:
            if raise_on_fail:
                raise
            return None

        allowed = self._allowed_symbols_by_position(length, constraints)
        state = initial_state
        sequence: list[int] = []
        for position in range(length):
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
            state = edge.dst

        decoded = [self.vocabulary.decode(symbol) for symbol in sequence]
        if not self.sequence_satisfies_constraints(decoded, constraints):
            if raise_on_fail:
                raise NoFeasibleSequenceError("Sampled sequence violates constraints.")
            return None
        return decoded

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
            encoded_values = {self.vocabulary.encode(self.get_viewpoint(value)) for value in values}
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
