from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Any, Hashable, Literal

from vo_regular_bp import (
    ConstraintSet,
    LongestFeasiblePolicy,
    OrderStackModel,
    SingletonAvoidingBackoffPolicy,
    SymbolTransform,
    VirtualAugmentedOrderStackModel,
    prepare_constrained_order_stack,
    prepare_until_end_order_stack,
    prepare_until_order_stack,
)
from vo_regular_bp.order_stack_bp import OrderSampleStep

from ctor.belief_propag import NoSolutionErrorInBP
from ctor.constraints import ConstraintProblem
from ctor.midi import MidiContinuatorBase, MidiRealizationStore
from ctor.vo_regular_bp.realization_store import TransformAwareMidiRealizationStore
from midi_stuff.mini_muse import Note


AugmentationMode = Literal["explicit", "virtual"]


@dataclass(frozen=True)
class ContinuatorTransform:
    """Pair a symbolic viewpoint transform with its MIDI-note realization."""

    symbol_transform: SymbolTransform
    apply_note: Callable[[Note], Note]


class _Start_vp:
    pass


class _End_vp:
    pass


def make_transposition_transforms(
    offsets: Iterable[int],
    *,
    fixed_symbols: Iterable[object] = (),
) -> tuple[ContinuatorTransform, ...]:
    fixed = frozenset(fixed_symbols)
    transforms = []
    for offset in offsets:
        shift = int(offset)

        def apply_symbol(symbol, shift=shift):
            if symbol in fixed:
                return symbol
            pitch, *rest = symbol
            return (pitch + shift, *rest)

        def inverse_symbol(symbol, shift=shift):
            if symbol in fixed:
                return symbol
            pitch, *rest = symbol
            return (pitch - shift, *rest)

        transforms.append(
            ContinuatorTransform(
                symbol_transform=SymbolTransform(
                    name=f"transpose_{shift:+d}",
                    apply_symbol=apply_symbol,
                    inverse_symbol=inverse_symbol,
                ),
                apply_note=lambda note, shift=shift: note.transpose(shift),
            )
        )
    return tuple(transforms)


class VORegularBPContinuator(MidiContinuatorBase):
    """
    Experimental MIDI Continuator using `vo_regular_bp` for generation.

    The symbolic model is rebuilt from learned viewpoint sequences after memory
    changes. MIDI realization remains separate, as in `ContextBPContinuator`.
    Explicit augmentation materializes transformed note sequences. Virtual
    augmentation keeps base note sequences and realizes transformed viewpoints
    lazily through a transform-aware realization store.
    """

    def __init__(
        self,
        midi_file: object = None,
        kmax: int = 4,
        transposition: bool = False,
        policy=None,
        augmentation_mode: AugmentationMode = "explicit",
        transposition_offsets: Iterable[int] | None = None,
        virtual_transposition: bool = False,
    ) -> None:
        self.kmax = int(kmax)
        self.policy = policy or SingletonAvoidingBackoffPolicy()
        self.start_symbol = _Start_vp()
        self.end_symbol = _End_vp()
        self.augmentation_mode: AugmentationMode = "virtual" if virtual_transposition else augmentation_mode
        if self.augmentation_mode not in {"explicit", "virtual"}:
            raise ValueError("augmentation_mode must be 'explicit' or 'virtual'")
        self.transposition_offsets = tuple(
            range(-6, 6) if transposition_offsets is None else transposition_offsets
        )
        self._last_generation_trace: list[dict[str, Any]] = []
        self._last_generation_diagnostics: dict[str, Any] = {}
        self.rng = random.Random(0)

        self.virtual_transforms = self._make_transforms(
            transposition if self.augmentation_mode == "virtual" else False
        )
        self.realization_store = self._new_realization_store()
        self.order_model = self._new_order_model()
        self.initialize_midi_state(transposition)
        if midi_file is not None:
            self.learn_file(midi_file, transposition)

    def _make_transforms(self, transposition: bool) -> tuple[ContinuatorTransform, ...]:
        offsets = self.transposition_offsets if transposition else (0,)
        return make_transposition_transforms(
            offsets,
            fixed_symbols=(self.start_symbol, self.end_symbol),
        )

    def _new_realization_store(self):
        if self.augmentation_mode == "virtual" and len(self.virtual_transforms) > 1:
            return TransformAwareMidiRealizationStore(
                self.get_viewpoint,
                transforms=self.virtual_transforms,
                start_padding=self.start_symbol,
                end_padding=self.end_symbol,
            )
        return MidiRealizationStore(
            vp_lambda=self.get_viewpoint,
            start_padding=self.start_symbol,
            end_padding=self.end_symbol,
        )

    def _new_order_model(self):
        sequences = [
            tuple(self.get_viewpoint(note) for note in sequence)
            for sequence in self.realization_store.input_sequences
        ]
        if not sequences:
            return OrderStackModel.from_sequences(
                [],
                max_order=max(1, self.kmax),
                start_symbol=self.start_symbol,
                end_symbol=self.end_symbol,
            )
        if self.augmentation_mode == "virtual" and len(self.virtual_transforms) > 1:
            return VirtualAugmentedOrderStackModel.from_sequences(
                sequences,
                max_order=max(1, self.kmax),
                transforms=[transform.symbol_transform for transform in self.virtual_transforms],
                start_symbol=self.start_symbol,
                end_symbol=self.end_symbol,
            )
        return OrderStackModel.from_sequences(
            sequences,
            max_order=max(1, self.kmax),
            start_symbol=self.start_symbol,
            end_symbol=self.end_symbol,
        )

    def _rebuild_order_model(self) -> None:
        self.order_model = self._new_order_model()

    def _relearn_sequences(self, sequences: list[list[object]]) -> None:
        self.realization_store = self._new_realization_store()
        for sequence in sequences:
            self.realization_store.learn_sequence(sequence)
        self._rebuild_order_model()

    def clear_memory(self):
        self.realization_store = self._new_realization_store()
        self._rebuild_order_model()

    def clear_first_n_phrases(self, n):
        self._relearn_sequences([list(sequence) for sequence in self.realization_store.input_sequences[n:]])

    def clear_last_phrase(self):
        if not self.realization_store.input_sequences:
            print("nothing to remove, memory is empty")
            return
        self._relearn_sequences([list(sequence) for sequence in self.realization_store.input_sequences[:-1]])

    def set_decay_mode(self, choice):
        """Compatibility no-op for the classic decay-mode control."""
        self.decay_mode = choice

    def sample_sequence_0(self, length=50, constraints=None):
        return self.sample_sequence(length=length, constraints=constraints)

    def learn_phrase(self, note_sequence, transposition):
        if len(note_sequence) == 0:
            return
        if self.forget_past and self.keep_last_n_melodies <= len(self.realization_store.input_sequences):
            self.clear_first_n_phrases(
                1 + len(self.realization_store.input_sequences) - self.keep_last_n_melodies
            )

        if self.augmentation_mode == "virtual" and len(self.virtual_transforms) > 1:
            self.realization_store.learn_sequence(list(note_sequence))
            self._rebuild_order_model()
            return
        if self.augmentation_mode == "virtual" and transposition:
            self._enable_virtual_transposition()
            self.realization_store.learn_sequence(list(note_sequence))
            self._rebuild_order_model()
            return

        trange = range(0, 1)
        if transposition:
            trange = self.transposition_offsets
        for t in trange:
            self.realization_store.learn_sequence(self.transpose_notes(note_sequence, int(t)))
        self._rebuild_order_model()

    def _enable_virtual_transposition(self) -> None:
        if len(self.virtual_transforms) > 1:
            return
        learned_sequences = [list(sequence) for sequence in self.realization_store.input_sequences]
        self.virtual_transforms = self._make_transforms(True)
        self.realization_store = self._new_realization_store()
        for sequence in learned_sequences:
            self.realization_store.learn_sequence(sequence)

    def get_start_vp(self):
        return self.start_symbol

    def get_end_vp(self):
        return self.end_symbol

    def sample_sequence(
        self,
        prefix=None,
        length=50,
        constraints=None,
        start_vp=None,
        relax_prefix_on_fail=True,
        relax_pos0_on_fail=True,
        raise_on_fail=False,
    ):
        effective_prefix = [start_vp] if start_vp is not None else prefix
        prefix_vps = self._prefix_viewpoints(effective_prefix)
        return self._sample_fixed(
            prefix_vps=prefix_vps,
            length=length,
            constraints=constraints,
            raise_on_fail=raise_on_fail,
        )

    def continue_sequence(
        self,
        prefix,
        length=50,
        constraints=None,
        relax_prefix_on_fail=True,
        relax_pos0_on_fail=True,
        raise_on_fail=False,
    ):
        return self._sample_fixed(
            prefix_vps=self._prefix_viewpoints(prefix),
            length=length,
            constraints=constraints,
            raise_on_fail=raise_on_fail,
        )

    def continue_until(
        self,
        prefix=None,
        *,
        stop,
        min_length=1,
        max_length=64,
        constraints=None,
        raise_on_fail=False,
    ):
        stop_condition = self._stop_condition(stop)
        try:
            backend = prepare_until_order_stack(
                self.order_model,
                prefix=tuple(self._prefix_viewpoints(prefix)),
                stop=stop_condition,
                min_length=min_length,
                max_length=max_length,
                constraints=self._to_constraint_set(constraints),
                policy=self.policy,
            )
            sequence, trace = backend.sample_with_trace(rng=self.rng)
        except ValueError as e:
            self._last_generation_trace = []
            self._last_generation_diagnostics = {}
            if raise_on_fail:
                raise NoSolutionErrorInBP("No solution for continue_until constraints.") from e
            return None
        self._record_generation(trace, backend.diagnostics.as_dict())
        return list(sequence)

    def continue_until_end(self, prefix=None, min_length=1, max_length=64, end_vp=None):
        end_symbol = self.end_symbol if end_vp is None else end_vp
        try:
            backend = prepare_until_end_order_stack(
                self.order_model,
                prefix=tuple(self._prefix_viewpoints(prefix)),
                end_symbol=self._as_viewpoint(end_symbol),
                min_length=min_length,
                max_length=max_length,
                policy=self.policy,
            )
            sequence, trace = backend.sample_with_trace(rng=self.rng)
        except ValueError:
            self._last_generation_trace = []
            self._last_generation_diagnostics = {}
            return None
        self._record_generation(trace, backend.diagnostics.as_dict())
        return list(sequence)

    def get_last_generation_trace(self) -> list[dict[str, Any]]:
        return list(self._last_generation_trace)

    def get_last_generation_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_generation_diagnostics)

    def _sample_fixed(
        self,
        *,
        prefix_vps: Sequence[object],
        length: int,
        constraints,
        raise_on_fail: bool,
    ):
        constraint_set = self._to_constraint_set(constraints)
        try:
            if self._final_position_accepts_end(constraint_set, length):
                backend = prepare_until_end_order_stack(
                    self.order_model,
                    prefix=tuple(prefix_vps),
                    end_symbol=self.end_symbol,
                    min_length=length,
                    max_length=length,
                    constraints=constraint_set,
                    policy=self.policy,
                )
            else:
                backend = prepare_constrained_order_stack(
                    self.order_model,
                    constraint_set,
                    length=length,
                    prefix=tuple(prefix_vps),
                    policy=self.policy,
                )
            sequence, trace = backend.sample_with_trace(rng=self.rng)
        except ValueError as e:
            self._last_generation_trace = []
            self._last_generation_diagnostics = {}
            if raise_on_fail:
                raise NoSolutionErrorInBP("No solution for constraints.") from e
            return None
        self._record_generation(trace, backend.diagnostics.as_dict())
        return list(sequence)

    def _prefix_viewpoints(self, prefix) -> list[object]:
        if prefix is None:
            return [self.start_symbol]
        viewpoints = [self._as_viewpoint(item) for item in prefix]
        return viewpoints or [self.start_symbol]

    def _as_viewpoint(self, value):
        if value is self.start_symbol or value is self.end_symbol:
            return value
        if hasattr(value, "pitch") and hasattr(value, "duration"):
            return self.get_viewpoint(value)
        return value

    def _stop_condition(self, stop):
        if callable(stop):
            return lambda symbol: bool(stop(self._as_viewpoint(symbol)))
        if isinstance(stop, (str, bytes)):
            return stop
        if self._is_known_symbol(stop):
            return self._as_viewpoint(stop)
        try:
            values = frozenset(stop)
        except TypeError:
            return self._as_viewpoint(stop)
        return frozenset(self._as_viewpoint(value) for value in values)

    def _is_known_symbol(self, value: object) -> bool:
        if not isinstance(value, Hashable):
            return False
        try:
            return value in self.order_model.alphabet
        except TypeError:
            return False

    def _to_constraint_set(self, constraints) -> ConstraintSet | None:
        if constraints is None:
            return None
        if isinstance(constraints, ConstraintSet):
            return constraints
        if isinstance(constraints, ConstraintProblem):
            return ConstraintSet(
                positional={
                    position: {self._as_viewpoint(value) for value in values}
                    for position, values in constraints.allowed_values_by_position.items()
                }
            )
        if isinstance(constraints, Mapping):
            return ConstraintSet(
                positional={
                    int(position): {self._as_viewpoint(value)}
                    for position, value in constraints.items()
                }
            )
        raise TypeError("constraints must be None, ConstraintProblem, ConstraintSet, or a mapping")

    def _final_position_accepts_end(self, constraints: ConstraintSet | None, length: int) -> bool:
        if constraints is None or length <= 0:
            return False
        final_constraint = constraints.positional.get(length - 1)
        if final_constraint is None:
            return False
        try:
            if callable(final_constraint):
                return bool(final_constraint(self.end_symbol))
            return self.end_symbol in final_constraint
        except (TypeError, ValueError):
            return False

    def _record_generation(self, trace: Sequence[OrderSampleStep], diagnostics: dict[str, Any]) -> None:
        self._last_generation_trace = [
            {
                "position": step.position,
                "symbol": self._trace_value(step.symbol),
                "order": step.order,
                "effective_order": step.order,
                "context": [self._trace_value(symbol) for symbol in step.context],
                "policy": step.policy,
                "candidate_orders": list(step.candidate_orders),
                "candidate_counts": list(step.candidate_counts),
                "skipped_orders": list(step.skipped_orders),
                "skipped_symbol": self._trace_value(step.skipped_symbol),
                "accepted_singleton": step.accepted_singleton,
                "suppressed_skipped_symbol": step.suppressed_skipped_symbol,
            }
            for step in trace
        ]
        self._last_generation_diagnostics = diagnostics

    @staticmethod
    def _trace_value(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, tuple):
            return [VORegularBPContinuator._trace_value(item) for item in value]
        return repr(value)

    def _realizable_viewpoint_sequence(self, vp_seq):
        return [
            vp
            for vp in vp_seq
            if vp is not self.start_symbol and vp is not self.end_symbol
        ]


__all__ = [
    "ContinuatorTransform",
    "VORegularBPContinuator",
    "make_transposition_transforms",
    "LongestFeasiblePolicy",
    "SingletonAvoidingBackoffPolicy",
]
