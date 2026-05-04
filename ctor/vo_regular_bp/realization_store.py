from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Hashable, Protocol

from midi_stuff.mini_muse import Note

from ctor.midi import MidiRealizationStore


class NoteTransform(Protocol):
    symbol_transform: object

    def apply_note(self, note: Note) -> Note:
        ...


class TransformAwareMidiRealizationStore(MidiRealizationStore):
    """
    MIDI realization memory for virtual symbolic augmentation.

    The store keeps only the original learned note sequences. For each learned
    note it indexes transformed viewpoints and stores an address carrying the
    transform id. Realization then lazily applies the transform to a copied
    base note.
    """

    def __init__(
        self,
        vp_lambda: Callable[[object], Hashable],
        *,
        transforms: Iterable[NoteTransform],
        start_padding: object | None = None,
        end_padding: object | None = None,
    ) -> None:
        self.transforms = tuple(transforms)
        if not self.transforms:
            raise ValueError("TransformAwareMidiRealizationStore requires at least one transform")
        super().__init__(
            vp_lambda,
            start_padding=start_padding,
            end_padding=end_padding,
        )

    def clear_memory(self) -> None:
        self.input_sequences: list[list[object]] = []
        self.viewpoints_realizations = defaultdict(list)

    def learn_sequence(self, sequence_of_stuff: Iterable[object]) -> None:
        material = list(sequence_of_stuff)
        sequence_index = len(self.input_sequences)
        self.input_sequences.append(material)
        for item_index, item in enumerate(material):
            viewpoint = self.get_viewpoint(item)
            for transform_index, transform in enumerate(self.transforms):
                transformed_viewpoint = transform.symbol_transform.apply_symbol(viewpoint)
                self.viewpoints_realizations[transformed_viewpoint].append(
                    (sequence_index, item_index, transform_index)
                )

    def get_input_object(self, obj_address: tuple[int, int] | tuple[int, int, int]) -> object:
        note = self.input_sequences[obj_address[0]][obj_address[1]]
        if len(obj_address) == 2:
            return note
        transform = self.transforms[obj_address[2]]
        return transform.apply_note(note)

    @staticmethod
    def is_starting_address(note_address: tuple[int, int] | tuple[int, int, int]) -> bool:
        return note_address[1] == 0

    def is_ending_address(self, note_address: tuple[int, int] | tuple[int, int, int]) -> bool:
        return note_address[1] == len(self.input_sequences[note_address[0]]) - 1
