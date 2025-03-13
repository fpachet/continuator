from __future__ import annotations
from pathlib import Path
from typing import Iterable, Any

import mido


def extract_pitches_from_midi(midi_file: Path | str) -> list[int]:
    """
    Extracts MIDI note sequence from a MIDI file.
    """
    mid = mido.MidiFile(midi_file)
    track = mid.tracks[min(1, len(mid.tracks) - 1)]
    return [msg.note for msg in track if msg.type == "note_on" and msg.velocity > 0]


def save_midi(sequence, output_file: Path | str = "generated.mid"):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for note in sequence:
        track.append(mido.Message("note_on", note=note, velocity=64, time=0))
        track.append(mido.Message("note_off", note=note, velocity=64, time=120))
    mid.save(output_file)


class MidiPitchCorpus:
    raw_note_seq: list[int]
    note_seq: list[int]
    index_seq: list[int]
    voc_size: int
    notes_to_idx: dict[int, int]
    idx_to_note: dict[int, int]
    shifts: list[int]

    def __init__(
        self,
        midi_file: Path | str,
        *,
        max_length: int = 0,
        pitch_shifts: Iterable[int] = (0,),
    ):
        self.note_seq = []
        self.shifts = list(pitch_shifts)
        self.raw_note_seq = extract_pitches_from_midi(midi_file)
        if max_length > 0:
            self.raw_note_seq = self.raw_note_seq[:max_length]
        for shift in self.shifts:
            self.note_seq.extend([note + shift for note in self.raw_note_seq])
        _unique_notes = list(set(self.note_seq))
        _unique_notes.sort()
        self.voc_size = len(_unique_notes)
        self.idx_to_note = dict(enumerate(_unique_notes))
        self.notes_to_idx = {v: k for k, v in self.idx_to_note.items()}
        self.index_seq = self.notes_to_indices(self.note_seq)

    def notes_to_indices(self, notes: list[int]) -> list[int]:
        return [self.notes_to_idx[note] for note in notes]

    def indices_to_notes(self, indices: Iterable[int]) -> list[int]:
        return [self.idx_to_note[idx] for idx in indices]

    def info(self) -> dict[str, Any]:
        return {
            "voc_size": self.voc_size,
            "raw_note_count": len(self.raw_note_seq),
            "note_count": len(self.note_seq),
            "index_seq": self.index_seq,
            "shifts": self.shifts,
        }
