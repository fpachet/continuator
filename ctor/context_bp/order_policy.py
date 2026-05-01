from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from ctor.context_bp.context_graph import ContextEdge, ContextGraph


@dataclass(frozen=True)
class CandidateSet:
    """Feasible outgoing edges for one attempted context order."""

    order: int
    graph: ContextGraph
    state: int
    edges: tuple[ContextEdge, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class CandidateChoice:
    candidate_set: CandidateSet
    edge: ContextEdge


class OrderPolicy(Protocol):
    """Choose one edge from ordered feasible context candidates."""

    def choose(
        self,
        candidate_sets: list[CandidateSet],
        rng: random.Random,
    ) -> CandidateChoice | None:
        ...


@dataclass(frozen=True)
class LongestFeasiblePolicy:
    """Use the highest-order feasible context."""

    def choose(
        self,
        candidate_sets: list[CandidateSet],
        rng: random.Random,
    ) -> CandidateChoice | None:
        for candidate_set in candidate_sets:
            edge = rng.choices(candidate_set.edges, weights=candidate_set.weights, k=1)[0]
            return CandidateChoice(candidate_set, edge)
        return None


@dataclass(frozen=True)
class SingletonAvoidingBackoffPolicy:
    """
    Classic Continuator-style singleton avoidance.

    A singleton higher-order continuation is accepted with probability
    `1 / (order + 1)`. Otherwise, the sampler backs off and suppresses that
    singleton symbol at intermediate orders when alternatives exist. Order 1 is
    left untouched so generation can still proceed when the singleton is forced.
    """

    def singleton_acceptance_probability(self, order: int) -> float:
        return 1.0 / (order + 1)

    def choose(
        self,
        candidate_sets: list[CandidateSet],
        rng: random.Random,
    ) -> CandidateChoice | None:
        skipped_symbol = None
        for candidate_set in candidate_sets:
            edges = candidate_set.edges
            weights = candidate_set.weights

            if skipped_symbol is not None and candidate_set.order > 1:
                filtered = [
                    (edge, weight)
                    for edge, weight in zip(edges, weights)
                    if edge.symbol != skipped_symbol
                ]
                if not filtered:
                    continue
                edges = tuple(edge for edge, _ in filtered)
                weights = tuple(weight for _, weight in filtered)

            if len(edges) == 1 and candidate_set.order > 1:
                if rng.random() > self.singleton_acceptance_probability(candidate_set.order):
                    skipped_symbol = edges[0].symbol
                    continue
                skipped_symbol = None

            edge = rng.choices(edges, weights=weights, k=1)[0]
            return CandidateChoice(candidate_set, edge)
        return None
