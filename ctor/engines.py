from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Mapping

from ctor.classic import Variable_order_Markov
from ctor.constraints import ConstraintProblem
from ctor.context_bp import ContextBPModel

EngineKind = Literal["classic", "context_bp"]
Constraints = ConstraintProblem | Mapping[int, Any] | None


class ClassicSequenceEngine:
    """Thin generic adapter around the current classic engine."""

    kind = "classic"

    def __init__(
        self,
        *,
        kmax: int = 5,
        viewpoint_fn: Callable[[Any], Any] | None = None,
        seed: int | None = None,
    ):
        self.model = Variable_order_Markov(None, viewpoint_fn, kmax=kmax, seed=seed)

    @property
    def start_symbol(self):
        return self.model.start_padding

    @property
    def end_symbol(self):
        return self.model.end_padding

    def learn_sequence(self, sequence: Iterable[Any]) -> None:
        self.model.learn_sequence(list(sequence))

    def sample_sequence(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None = None,
        constraints: Constraints = None,
        raise_on_fail: bool = False,
    ) -> list[Any] | None:
        return self.model.sample_sequence(
            length=length,
            prefix=list(prefix) if prefix is not None else None,
            constraints=constraints,
            raise_on_fail=raise_on_fail,
        )

    def continue_sequence(
        self,
        prefix: Iterable[Any],
        *,
        length: int,
        constraints: Constraints = None,
        raise_on_fail: bool = False,
    ) -> list[Any] | None:
        return self.model.continue_sequence(
            list(prefix),
            length=length,
            constraints=constraints,
            raise_on_fail=raise_on_fail,
        )

    def continue_until_end(
        self,
        prefix: Iterable[Any] | None = None,
        *,
        min_length: int = 1,
        max_length: int = 64,
        end_symbol: Any | None = None,
    ) -> list[Any] | None:
        return self.model.continue_until_end(
            prefix=list(prefix) if prefix is not None else None,
            min_length=min_length,
            max_length=max_length,
            end_vp=end_symbol,
        )


class ContextBPSequenceEngine:
    """Thin generic adapter around the experimental context-BP engine."""

    kind = "context_bp"

    def __init__(
        self,
        *,
        kmax: int = 5,
        viewpoint_fn: Callable[[Any], Any] | None = None,
        seed: int | None = None,
    ):
        self.model = ContextBPModel(kmax=kmax, viewpoint_fn=viewpoint_fn, seed=seed)

    @property
    def start_symbol(self):
        return self.model.start_symbol

    @property
    def end_symbol(self):
        return self.model.end_symbol

    def learn_sequence(self, sequence: Iterable[Any]) -> None:
        self.model.learn_sequence(list(sequence))

    def sample_sequence(
        self,
        length: int,
        *,
        prefix: Iterable[Any] | None = None,
        constraints: Constraints = None,
        raise_on_fail: bool = False,
    ) -> list[Any] | None:
        return self.model.sample_sequence(
            length=length,
            prefix=list(prefix) if prefix is not None else None,
            constraints=constraints,
            raise_on_fail=raise_on_fail,
        )

    def continue_sequence(
        self,
        prefix: Iterable[Any],
        *,
        length: int,
        constraints: Constraints = None,
        raise_on_fail: bool = False,
    ) -> list[Any] | None:
        return self.model.sample_sequence(
            length=length,
            prefix=list(prefix),
            constraints=constraints,
            raise_on_fail=raise_on_fail,
        )

    def continue_until_end(
        self,
        prefix: Iterable[Any] | None = None,
        *,
        min_length: int = 1,
        max_length: int = 64,
        end_symbol: Any | None = None,
    ) -> list[Any] | None:
        return self.model.continue_until_end(
            prefix=list(prefix) if prefix is not None else None,
            min_length=min_length,
            max_length=max_length,
            end_symbol=end_symbol,
        )


def make_sequence_engine(
    kind: EngineKind | str = "classic",
    *,
    kmax: int = 5,
    viewpoint_fn: Callable[[Any], Any] | None = None,
    seed: int | None = None,
) -> ClassicSequenceEngine | ContextBPSequenceEngine:
    """Create a generic sequence engine without changing the MIDI facade."""
    normalized = kind.replace("-", "_")
    if normalized == "classic":
        return ClassicSequenceEngine(kmax=kmax, viewpoint_fn=viewpoint_fn, seed=seed)
    if normalized in {"context_bp", "new"}:
        return ContextBPSequenceEngine(kmax=kmax, viewpoint_fn=viewpoint_fn, seed=seed)
    raise ValueError(f"unknown sequence engine: {kind!r}")


__all__ = [
    "ClassicSequenceEngine",
    "ContextBPSequenceEngine",
    "EngineKind",
    "make_sequence_engine",
]
