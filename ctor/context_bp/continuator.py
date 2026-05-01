from __future__ import annotations

from typing import Any

from ctor.context_bp.model import ContextBPModel
from ctor.midi import MidiContinuatorBase
from ctor.classic.variable_order_markov import Variable_order_Markov


class ContextBPContinuator(MidiContinuatorBase):
    """
    Experimental MIDI Continuator using context-BP for viewpoint generation.

    The classic `Variable_order_Markov` store is maintained only as a MIDI
    realization memory. This class deliberately does not inherit from
    `Continuator2`; both facades use shared MIDI utilities instead.
    """

    def __init__(self, midi_file: object = None, kmax: int = 4, transposition: bool = False) -> None:
        self.kmax = int(kmax)
        self.vom = self._new_realization_store()
        self.context_model = self._new_context_model()
        self.initialize_midi_state(transposition)
        if midi_file is not None:
            self.learn_file(midi_file, transposition)

    def _new_realization_store(self) -> Variable_order_Markov:
        return Variable_order_Markov(
            sequence_of_stuff=None,
            vp_lambda=self.get_viewpoint,
            kmax=self.kmax,
            seed=0,
        )

    def _new_context_model(self) -> ContextBPModel:
        return ContextBPModel(kmax=self.kmax, viewpoint_fn=self.get_viewpoint, seed=0)

    def _relearn_sequences(self, sequences: list[list[Any]]) -> None:
        self.vom = self._new_realization_store()
        self.context_model = self._new_context_model()
        for sequence in sequences:
            material = list(sequence)
            self.vom.learn_sequence(material)
            self.context_model.learn_sequence(material)

    def clear_memory(self):
        self.vom = self._new_realization_store()
        self.context_model = self._new_context_model()

    def clear_first_n_phrases(self, n):
        self._relearn_sequences([list(sequence) for sequence in self.vom.input_sequences[n:]])

    def clear_last_phrase(self):
        if not self.vom.input_sequences:
            print("nothing to remove, memory is empty")
            return
        self._relearn_sequences([list(sequence) for sequence in self.vom.input_sequences[:-1]])

    def learn_phrase(self, note_sequence, transposition):
        if len(note_sequence) == 0:
            return
        if self.forget_past and self.keep_last_n_melodies <= len(self.vom.input_sequences):
            self.clear_first_n_phrases(1 + len(self.vom.input_sequences) - self.keep_last_n_melodies)

        trange = range(0, 1)
        if transposition:
            trange = range(-6, 6, 1)
        for t in trange:
            transposed = self.transpose_notes(note_sequence, t)
            self.vom.learn_sequence(transposed)
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
        return self.context_model.sample_sequence(
            length=length,
            prefix=effective_prefix,
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

    def _realizable_viewpoint_sequence(self, vp_seq):
        return [
            vp
            for vp in vp_seq
            if vp is not self.context_model.start_symbol and vp is not self.context_model.end_symbol
        ]
