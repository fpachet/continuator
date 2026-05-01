from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextEdge:
    src: int
    dst: int
    symbol: int
    weight: float
    order: int


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

    def continuation_distribution_with_order(
        self,
        context: tuple[int, ...],
    ) -> tuple[dict[int, float], int | None]:
        suffix = self.longest_available_suffix(context)
        if suffix is None:
            return {}, None
        counts = self.counts[suffix]
        total = sum(counts.values())
        if total <= 0:
            return {}, None
        return {symbol: count / total for symbol, count in counts.items()}, len(suffix)


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
        max_order: int | None = None,
    ) -> "ContextGraph":
        if max_order is None:
            max_order = counts.kmax
        if max_order < 1 or max_order > counts.kmax:
            raise ValueError(f"max_order must be between 1 and {counts.kmax}")

        graph = cls(max_order)
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
            if len(context) <= graph.kmax:
                add_context(context)
        for context in initial_contexts or []:
            add_context(context)

        while queue:
            context = queue.popleft()
            src = graph.context_to_id[context]
            distribution, order = counts.continuation_distribution_with_order(context)
            for symbol, weight in distribution.items():
                dst_context = graph.next_context(context, symbol)
                dst = add_context(dst_context)
                graph.outgoing[src].append(
                    ContextEdge(
                        src=src,
                        dst=dst,
                        symbol=symbol,
                        weight=float(weight),
                        order=order or 0,
                    )
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

    def first_hit_reachable_to_symbol(self, target_symbol: int, max_steps: int) -> list[set[int]]:
        """
        Return states that can first emit `target_symbol` in exactly n steps.

        The result at index `n` contains source state ids. Index 0 is always
        empty because first hit is defined over emitted symbols.
        """
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        reachable: list[set[int]] = [set() for _ in range(max_steps + 1)]
        for steps in range(1, max_steps + 1):
            states = reachable[steps]
            for state, edges in enumerate(self.outgoing):
                for edge in edges:
                    if steps == 1:
                        if edge.symbol == target_symbol:
                            states.add(state)
                            break
                    elif edge.symbol != target_symbol and edge.dst in reachable[steps - 1]:
                        states.add(state)
                        break
        return reachable

    @staticmethod
    def can_reach_between(
        reachable: list[set[int]],
        state: int,
        min_steps: int,
        max_steps: int,
    ) -> bool:
        """Return whether `state` can first hit the target inside the window."""
        if min_steps < 0:
            raise ValueError("min_steps must be non-negative")
        if max_steps < min_steps:
            return False
        capped_max = min(max_steps, len(reachable) - 1)
        return any(state in reachable[steps] for steps in range(min_steps, capped_max + 1))
