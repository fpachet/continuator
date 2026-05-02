from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, Iterable


class MidiRealizationStore:
    """Minimal viewpoint-to-note memory used by MIDI facades."""

    def __init__(
        self,
        vp_lambda: Callable[[object], Hashable],
        *,
        start_padding: object | None = None,
        end_padding: object | None = None,
    ) -> None:
        self.viewpoint_lambda = vp_lambda
        self.start_padding = object() if start_padding is None else start_padding
        self.end_padding = object() if end_padding is None else end_padding
        self.clear_memory()

    def clear_memory(self) -> None:
        self.input_sequences: list[list[object]] = []
        self.viewpoints_realizations = defaultdict(list)

    def clear_first_N_phrases(self, n: int) -> None:
        if not self.input_sequences:
            print("nothing to remove, memory is empty")
            return
        if len(self.input_sequences) < n:
            print("nothing to remove, memory is less than " + str(n))
            return
        sequences_to_learn = self.input_sequences[n:]
        self.clear_memory()
        for sequence in sequences_to_learn:
            self.learn_sequence(sequence)

    def clear_last_phrase(self) -> None:
        if not self.input_sequences:
            print("nothing to remove, memory is empty")
            return
        sequences_to_learn = self.input_sequences[:-1]
        self.clear_memory()
        for sequence in sequences_to_learn:
            self.learn_sequence(sequence)

    def learn_sequence(self, sequence_of_stuff: Iterable[object]) -> None:
        material = list(sequence_of_stuff)
        sequence_index = len(self.input_sequences)
        self.input_sequences.append(material)
        for item_index, item in enumerate(material):
            viewpoint = self.get_viewpoint(item)
            self.viewpoints_realizations[viewpoint].append((sequence_index, item_index))

    def get_viewpoint(self, real_object: object) -> Hashable:
        return self.viewpoint_lambda(real_object)

    def get_input_object(self, obj_address: tuple[int, int]) -> object:
        return self.input_sequences[obj_address[0]][obj_address[1]]

    @staticmethod
    def is_starting_address(note_address: tuple[int, int]) -> bool:
        return note_address[1] == 0

    def is_ending_address(self, note_address: tuple[int, int]) -> bool:
        return note_address[1] == len(self.input_sequences[note_address[0]]) - 1
