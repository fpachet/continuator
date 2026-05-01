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
    decision: "PolicyDecision"


@dataclass(frozen=True)
class PolicyDecision:
    """Traceable explanation of an order-policy choice."""

    policy_name: str
    candidate_orders: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    selected_order: int | None = None
    selected_symbol: int | None = None
    skipped_orders: tuple[int, ...] = ()
    skipped_symbol: int | None = None
    accepted_singleton: bool = False
    suppressed_skipped_symbol: bool = False


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

    policy_name: str = "longest_feasible"

    def choose(
        self,
        candidate_sets: list[CandidateSet],
        rng: random.Random,
    ) -> CandidateChoice | None:
        for candidate_set in candidate_sets:
            edge = rng.choices(candidate_set.edges, weights=candidate_set.weights, k=1)[0]
            return CandidateChoice(
                candidate_set,
                edge,
                PolicyDecision(
                    policy_name=self.policy_name,
                    candidate_orders=tuple(candidate.order for candidate in candidate_sets),
                    candidate_counts=tuple(len(candidate.edges) for candidate in candidate_sets),
                    selected_order=candidate_set.order,
                    selected_symbol=edge.symbol,
                ),
            )
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

    acceptance_probability: float | None = None
    min_singleton_order: int = 2
    suppress_skipped_symbol: bool = True
    policy_name: str = "singleton_avoiding_backoff"

    def __post_init__(self) -> None:
        if self.acceptance_probability is not None:
            if self.acceptance_probability < 0 or self.acceptance_probability > 1:
                raise ValueError("acceptance_probability must be between 0 and 1")
        if self.min_singleton_order < 1:
            raise ValueError("min_singleton_order must be at least 1")

    def singleton_acceptance_probability(self, order: int) -> float:
        if self.acceptance_probability is not None:
            return self.acceptance_probability
        return 1.0 / (order + 1)

    def choose(
        self,
        candidate_sets: list[CandidateSet],
        rng: random.Random,
    ) -> CandidateChoice | None:
        skipped_symbol = None
        skipped_orders: list[int] = []
        suppressed = False
        candidate_orders = tuple(candidate.order for candidate in candidate_sets)
        candidate_counts = tuple(len(candidate.edges) for candidate in candidate_sets)
        for candidate_set in candidate_sets:
            edges = candidate_set.edges
            weights = candidate_set.weights

            if (
                self.suppress_skipped_symbol
                and skipped_symbol is not None
                and candidate_set.order >= self.min_singleton_order
            ):
                filtered = [
                    (edge, weight)
                    for edge, weight in zip(edges, weights)
                    if edge.symbol != skipped_symbol
                ]
                if not filtered:
                    suppressed = True
                    continue
                if len(filtered) < len(edges):
                    suppressed = True
                edges = tuple(edge for edge, _ in filtered)
                weights = tuple(weight for _, weight in filtered)

            if len(edges) == 1 and candidate_set.order >= self.min_singleton_order:
                if rng.random() > self.singleton_acceptance_probability(candidate_set.order):
                    skipped_symbol = edges[0].symbol
                    skipped_orders.append(candidate_set.order)
                    continue
                skipped_symbol = None

            edge = rng.choices(edges, weights=weights, k=1)[0]
            return CandidateChoice(
                candidate_set,
                edge,
                PolicyDecision(
                    policy_name=self.policy_name,
                    candidate_orders=candidate_orders,
                    candidate_counts=candidate_counts,
                    selected_order=candidate_set.order,
                    selected_symbol=edge.symbol,
                    skipped_orders=tuple(skipped_orders),
                    skipped_symbol=skipped_symbol,
                    accepted_singleton=len(edges) == 1 and candidate_set.order >= self.min_singleton_order,
                    suppressed_skipped_symbol=suppressed,
                ),
            )
        return None
