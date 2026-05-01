from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextEdge:
    src: int
    dst: int
    symbol: int
    weight: float


class ContextCounts:
    """Variable-order continuation counts over integer symbol ids."""

    def __init__(self, kmax: int):
        if kmax < 1:
            raise ValueError("kmax must be at least 1")
        self.kmax = int(kmax)
        self.counts: dict[tuple[int, ...], Counter[int]] = {}

    def update_sequence(self, sequence: list[int]) -> None:
        for next_position in range(1, len(sequence)):
            next_symbol = sequence[next_position]
            max_order = min(self.kmax, next_position)
            for order in range(1, max_order + 1):
                context = tuple(sequence[next_position - order:next_position])
                self.counts.setdefault(context, Counter())[next_symbol] += 1

    def longest_available_suffix(self, context: tuple[int, ...]) -> tuple[int, ...] | None:
        max_order = min(self.kmax, len(context))
        for order in range(max_order, 0, -1):
            suffix = tuple(context[-order:])
            if suffix in self.counts:
                return suffix
        return None

    def continuation_counts(self, context: tuple[int, ...]) -> Counter[int] | None:
        suffix = self.longest_available_suffix(context)
        if suffix is None:
            return None
        return self.counts[suffix]

    def continuation_distribution(self, context: tuple[int, ...]) -> dict[int, float]:
        counts = self.continuation_counts(context)
        if not counts:
            return {}
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {symbol: count / total for symbol, count in counts.items()}


class ContextGraph:
    """Sparse context-state graph compiled from variable-order counts."""

    def __init__(self, kmax: int):
        self.kmax = int(kmax)
        self.contexts: list[tuple[int, ...]] = []
        self.context_to_id: dict[tuple[int, ...], int] = {}
        self.outgoing: list[list[ContextEdge]] = []

    @classmethod
    def from_counts(
        cls,
        counts: ContextCounts,
        *,
        initial_contexts: list[tuple[int, ...]] | None = None,
    ) -> "ContextGraph":
        graph = cls(counts.kmax)
        queue: deque[tuple[int, ...]] = deque()

        def add_context(context: tuple[int, ...]) -> int:
            normalized = graph.truncate_context(context)
            if normalized in graph.context_to_id:
                return graph.context_to_id[normalized]
            context_id = len(graph.contexts)
            graph.context_to_id[normalized] = context_id
            graph.contexts.append(normalized)
            graph.outgoing.append([])
            queue.append(normalized)
            return context_id

        for context in counts.counts:
            add_context(context)
        for context in initial_contexts or []:
            add_context(context)

        while queue:
            context = queue.popleft()
            src = graph.context_to_id[context]
            distribution = counts.continuation_distribution(context)
            for symbol, weight in distribution.items():
                dst_context = graph.next_context(context, symbol)
                dst = add_context(dst_context)
                graph.outgoing[src].append(
                    ContextEdge(src=src, dst=dst, symbol=symbol, weight=float(weight))
                )

        return graph

    def truncate_context(self, context: tuple[int, ...]) -> tuple[int, ...]:
        if len(context) <= self.kmax:
            return tuple(context)
        return tuple(context[-self.kmax:])

    def next_context(self, context: tuple[int, ...], symbol: int) -> tuple[int, ...]:
        return self.truncate_context(tuple(context) + (symbol,))

    def state_id(self, context: tuple[int, ...]) -> int:
        normalized = self.truncate_context(context)
        try:
            return self.context_to_id[normalized]
        except KeyError as e:
            raise KeyError(f"Unknown context state: {normalized!r}") from e
