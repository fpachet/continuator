from __future__ import annotations

from typing import Any

from ctor.context_bp.model import ContextBPModel
from ctor.context_bp.order_policy import OrderPolicy, SingletonAvoidingBackoffPolicy
from ctor.midi import MidiContinuatorBase, MidiRealizationStore


class ContextBPContinuator(MidiContinuatorBase):
    """
    Experimental MIDI Continuator using context-BP for viewpoint generation.

    A lightweight MIDI realization store maps generated viewpoints back to
    learned note addresses. This class deliberately does not inherit from
    `Continuator2`; both facades use shared MIDI utilities instead.
    """

    def __init__(
        self,
        midi_file: object = None,
        kmax: int = 4,
        transposition: bool = False,
        order_policy: OrderPolicy | None = None,
    ) -> None:
        self.kmax = int(kmax)
        self.order_policy = order_policy or SingletonAvoidingBackoffPolicy()
        self.context_model = self._new_context_model()
        self.realization_store = self._new_realization_store()
        self.initialize_midi_state(transposition)
        if midi_file is not None:
            self.learn_file(midi_file, transposition)

    def _new_realization_store(self) -> MidiRealizationStore:
        return MidiRealizationStore(
            vp_lambda=self.get_viewpoint,
            start_padding=self.context_model.start_symbol,
            end_padding=self.context_model.end_symbol,
        )

    def _new_context_model(self) -> ContextBPModel:
        return ContextBPModel(
            kmax=self.kmax,
            viewpoint_fn=self.get_viewpoint,
            seed=0,
            order_policy=self.order_policy,
        )

    def _relearn_sequences(self, sequences: list[list[Any]]) -> None:
        self.context_model = self._new_context_model()
        self.realization_store = self._new_realization_store()
        for sequence in sequences:
            material = list(sequence)
            self.realization_store.learn_sequence(material)
            self.context_model.learn_sequence(material)

    def clear_memory(self):
        self.context_model = self._new_context_model()
        self.realization_store = self._new_realization_store()

    def clear_first_n_phrases(self, n):
        self._relearn_sequences([list(sequence) for sequence in self.realization_store.input_sequences[n:]])

    def clear_last_phrase(self):
        if not self.realization_store.input_sequences:
            print("nothing to remove, memory is empty")
            return
        self._relearn_sequences([list(sequence) for sequence in self.realization_store.input_sequences[:-1]])

    def learn_phrase(self, note_sequence, transposition):
        if len(note_sequence) == 0:
            return
        if self.forget_past and self.keep_last_n_melodies <= len(self.realization_store.input_sequences):
            self.clear_first_n_phrases(
                1 + len(self.realization_store.input_sequences) - self.keep_last_n_melodies
            )

        trange = range(0, 1)
        if transposition:
            trange = range(-6, 6, 1)
        for t in trange:
            transposed = self.transpose_notes(note_sequence, t)
            self.realization_store.learn_sequence(transposed)
            self.context_model.learn_sequence(transposed)

    def get_start_vp(self):
        return self.context_model.start_symbol

    def get_end_vp(self):
        return self.context_model.end_symbol

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
        initial_mode = "free" if effective_prefix is None else "start"
        return self.context_model.sample_sequence(
            length=length,
            prefix=effective_prefix,
            constraints=constraints,
            initial_mode=initial_mode,
            raise_on_fail=raise_on_fail,
        )

    def sample_sequence_with_trace(
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
        initial_mode = "free" if effective_prefix is None else "start"
        return self.context_model.sample_sequence_with_trace(
            length=length,
            prefix=effective_prefix,
            constraints=constraints,
            initial_mode=initial_mode,
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
        return self.context_model.sample_sequence(
            length=length,
            prefix=prefix,
            constraints=constraints,
            raise_on_fail=raise_on_fail,
        )

    def continue_until_end(self, prefix=None, min_length=1, max_length=64, end_vp=None):
        return self.context_model.continue_until_end(
            prefix=prefix,
            min_length=min_length,
            max_length=max_length,
            end_symbol=end_vp,
        )

    def continue_until_end_with_trace(
        self,
        prefix=None,
        min_length=1,
        max_length=64,
        end_vp=None,
        raise_on_fail=False,
    ):
        return self.context_model.continue_until_end_with_trace(
            prefix=prefix,
            min_length=min_length,
            max_length=max_length,
            end_symbol=end_vp,
            raise_on_fail=raise_on_fail,
        )

    def get_last_generation_trace(self) -> list[dict[str, Any]]:
        return self.context_model.last_sample_trace_as_dicts()

    def _realizable_viewpoint_sequence(self, vp_seq):
        return [
            vp
            for vp in vp_seq
            if vp is not self.context_model.start_symbol and vp is not self.context_model.end_symbol
        ]
