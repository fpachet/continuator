from __future__ import annotations

import os
import pathlib
import random
from difflib import SequenceMatcher
from typing import Iterable

import mido
import numpy as np

from midi_stuff.mini_muse import Note


class MidiContinuatorBase:
    """
    Shared MIDI utilities used by classic and context-BP facades.

    Concrete facades provide their own generation engine. This base class only
    handles MIDI parsing, phrase memory access, note realization, and MIDI file
    output.
    """

    def _midi_store(self):
        # Context-BP has a dedicated realization store; the classic facade uses
        # its Variable_order_Markov object as both model and realization store.
        if hasattr(self, "realization_store"):
            return self.realization_store
        return self.vom

    def initialize_midi_state(self, transposition: bool = False) -> None:
        self.learn_input = True
        self.tempo_msgs = []
        self.transpose = transposition
        self.forget_past = False
        self.keep_last_n_melodies = 20
        self.generate_length = 10

    @staticmethod
    def get_viewpoint(note):
        nb_beats_per_bin = 1
        return (
            note.pitch,
            int(note.duration / nb_beats_per_bin),
            note.overlaps_left(),
            note.overlaps_right(),
        )

    def set_learn_input(self, value):
        self.learn_input = value

    def get_learn_input(self):
        return self.learn_input

    def set_forget(self, forget_past):
        self.forget_past = forget_past

    def set_keep_last(self, keep):
        self.keep_last_n_melodies = keep

    def set_transpose(self, trans):
        self.transpose = trans

    def get_phrase_titles(self):
        return [f"{i + 1} phrase with {len(phrase)} notes" for i, phrase in enumerate(self._midi_store().input_sequences)]

    def get_phrase(self, index):
        return self._midi_store().input_sequences[index]

    def clear_memory(self):
        self._midi_store().clear_memory()

    def clear_first_n_phrases(self, n):
        self._midi_store().clear_first_N_phrases(n)

    def clear_last_phrase(self):
        self._midi_store().clear_last_phrase()

    def learn_file(self, midi_file, transposition):
        notes_original = self.extract_notes(midi_file)
        self.learn_phrase(notes_original, transposition)

    def learn_folder(self, folder_path, transpose=False):
        all_files = []
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname.lower().endswith((".mid", ".midi")):
                    full_path = os.path.join(root, fname)
                    all_files.append(full_path)
                    self.learn_file(full_path, transpose)
        return all_files

    def learn_files(self, files, transposition=False):
        for file in files:
            self.learn_file(file, transposition)

    def learn_phrase_from_mido(self, phrase):
        self.learn_phrase(self.get_phrase_from_mido(phrase), False)

    def get_phrase_from_mido(self, phrase):
        sequence = []
        pending_notes = {}
        start_time = 0
        for msg in phrase:
            start_time = start_time + msg.time
            msg.time = start_time
        for msg in phrase:
            if msg.type == "note_on" and msg.velocity > 0:
                pending_notes[msg.note] = msg
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note not in pending_notes:
                    print("problem: note off does not match previous note on: " + str(msg.note))
                else:
                    note_on_msg = pending_notes[msg.note]
                    start_time = note_on_msg.time * 2
                    duration = (msg.time - note_on_msg.time) * 2
                    sequence.append(Note(note_on_msg.note, note_on_msg.velocity, duration, start_time))
        self.set_delta_notes(sequence)
        return sequence

    @staticmethod
    def transpose_notes(notes, t):
        return [n.transpose(t) for n in notes]

    def get_input_note(self, note_address):
        return self._midi_store().get_input_object(note_address)

    def is_starting_address(self, note_address):
        return self._midi_store().is_starting_address(note_address)

    def is_ending_address(self, note_address):
        return self._midi_store().is_ending_address(note_address)

    def extract_notes(self, midi_file):
        """Extract the sequence of note-on events from a MIDI file."""
        mid = mido.MidiFile(midi_file)
        resolution = mid.ticks_per_beat
        notes = []
        pending_notes = np.empty(128, dtype=object)
        pending_start_times = np.zeros(128)
        current_time = 0
        for track in mid.tracks:
            for msg in track:
                current_time += 2 * mido.tick2second(msg.time, ticks_per_beat=resolution, tempo=500000)
                if msg.type == "set_tempo":
                    self.tempo_msgs.append(msg.tempo)
                if msg.type == "note_on" and msg.velocity > 0:
                    new_note = Note(msg.note, msg.velocity, 0)
                    notes.append(new_note)
                    pending_notes[msg.note] = new_note
                    pending_start_times[msg.note] = current_time
                    new_note.set_start_time(current_time)
                    new_note.set_duration(1)
                if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if pending_notes[msg.note] is None:
                        print("found 0 velocity note, skipping it")
                        continue
                    pending_note = pending_notes[msg.note]
                    pending_note.set_duration(current_time - pending_start_times[msg.note])
                    pending_notes[msg.note] = None
                    pending_start_times[msg.note] = 0
        self.set_delta_notes(notes)
        return np.array(notes)

    @staticmethod
    def set_delta_notes(notes):
        for i, note in enumerate(notes):
            note.overlaps_left_flag = False
            note.next_start_delta = 0
            if i > 0:
                note.overlaps_left_flag = note.start_time < notes[i - 1].get_end_time()
            if i < len(notes) - 1:
                note.next_start_delta = notes[i + 1].start_time - note.get_end_time()

    @staticmethod
    def all_midi_files_from_path(path_string):
        path = pathlib.Path(path_string)
        return list(path.glob("*.mid")) + list(path.glob("*.midi"))

    def _realizable_viewpoint_sequence(self, vp_seq: Iterable[object]) -> list[object]:
        midi_store = self._midi_store()
        start_padding = getattr(midi_store, "start_padding", None)
        end_padding = getattr(midi_store, "end_padding", None)
        return [
            viewpoint
            for viewpoint in vp_seq
            if viewpoint != start_padding and viewpoint != end_padding
        ]

    def realize_vp_sequence(self, vp_seq):
        raw_viewpoints = list(vp_seq)
        midi_store = self._midi_store()
        end_padding = getattr(midi_store, "end_padding", None)
        force_ending_realization = bool(raw_viewpoints and raw_viewpoints[-1] == end_padding)
        viewpoints = self._realizable_viewpoint_sequence(raw_viewpoints)
        if not viewpoints:
            return []

        domains = self._realization_domains(
            viewpoints,
            force_ending_realization=force_ending_realization,
        )
        note_addresses = self._best_realization_address_sequence(domains)
        return self.set_timing(note_addresses)

    def _realization_domains(self, viewpoints, *, force_ending_realization=False):
        midi_store = self._midi_store()
        domains = []
        for i, viewpoint in enumerate(viewpoints):
            realizations = list(midi_store.viewpoints_realizations.get(viewpoint, []))
            if not realizations:
                raise ValueError(f"No MIDI realization for viewpoint: {viewpoint!r}")

            starting_realizations = []
            if i == 0:
                starting_realizations = [
                    address for address in realizations if self.is_starting_address(address)
                ]
                if starting_realizations:
                    realizations = starting_realizations

            if i == len(viewpoints) - 1 and force_ending_realization:
                ending_realizations = [
                    address
                    for address in midi_store.viewpoints_realizations[viewpoint]
                    if self.is_ending_address(address)
                ]
                if i == 0 and starting_realizations:
                    start_and_end = [
                        address for address in starting_realizations if self.is_ending_address(address)
                    ]
                    if start_and_end:
                        ending_realizations = start_and_end
                if ending_realizations:
                    realizations = ending_realizations

            domains.append(realizations)
        return domains

    def _best_realization_address_sequence(self, domains):
        if not domains:
            return []
        if len(domains) == 1:
            return [domains[0][0]]

        features = [self._realization_domain_features(domain) for domain in domains]
        costs = np.zeros(len(domains[0]), dtype=np.float64)
        backpointers = []

        for position in range(len(domains) - 1):
            transition_costs = self._realization_transition_cost_matrix(
                features[position],
                features[position + 1],
            )
            total_costs = costs[:, None] + transition_costs
            best_previous = np.argmin(total_costs, axis=0)
            costs = total_costs[best_previous, np.arange(total_costs.shape[1])]
            backpointers.append(best_previous)

        best_index = int(np.argmin(costs))
        indices = [0] * len(domains)
        indices[-1] = best_index
        for position in range(len(domains) - 2, -1, -1):
            best_index = int(backpointers[position][best_index])
            indices[position] = best_index

        return [domains[position][index] for position, index in enumerate(indices)]

    def _realization_domain_features(self, domain):
        sequence_positions = [self._address_sequence_position(address) for address in domain]
        notes = [self.get_input_note(address) for address in domain]
        return {
            "sequence": np.array([sequence for sequence, _index in sequence_positions], dtype=np.int64),
            "index": np.array([index for _sequence, index in sequence_positions], dtype=np.int64),
            "transform": np.array(
                [self._address_transform_index(address) for address in domain],
                dtype=np.int64,
            ),
            "overlaps_left": np.array([bool(note.overlaps_left()) for note in notes], dtype=bool),
            "overlaps_right": np.array([bool(note.overlaps_right()) for note in notes], dtype=bool),
        }

    @staticmethod
    def _realization_transition_cost_matrix(left, right):
        same_sequence = (
            (left["sequence"][:, None] >= 0)
            & (left["sequence"][:, None] == right["sequence"][None, :])
        )
        gap = right["index"][None, :] - left["index"][:, None]
        same_sequence_cost = np.abs(gap - 1).astype(np.float64) * 10.0
        same_sequence_cost += (gap <= 0).astype(np.float64) * 20.0
        cost = np.where(same_sequence, same_sequence_cost, 100.0)

        incompatible_overlap = left["overlaps_right"][:, None] != right["overlaps_left"][None, :]
        cost += incompatible_overlap.astype(np.float64) * 10_000.0

        left_transform = left["transform"][:, None]
        right_transform = right["transform"][None, :]
        both_transformed = (left_transform >= 0) & (right_transform >= 0)
        transform_change = both_transformed & (left_transform != right_transform)
        cost += transform_change.astype(np.float64) * 5.0
        return cost

    @staticmethod
    def _address_sequence_position(address):
        try:
            return int(address[0]), int(address[1])
        except (IndexError, TypeError, ValueError):
            return -1, -1

    @staticmethod
    def _address_transform_index(address):
        try:
            return int(address[2])
        except (IndexError, TypeError, ValueError):
            return -1

    def get_vp_for_pitch(self, pitch):
        vps = []
        midi_store = self._midi_store()
        for vp, notes in midi_store.viewpoints_realizations.items():
            for note_address in notes:
                note = midi_store.get_input_object(note_address)
                if note.pitch == pitch:
                    vps.append(vp)
        return random.choice(vps)

    def set_timing(self, idx_sequence):
        sequence = []
        start_time = 0
        for i, note_address in enumerate(idx_sequence):
            note_copy = self.get_input_note(note_address).copy()
            if sequence:
                preceding = sequence[-1]
                preceding_address = idx_sequence[i - 1]
                delta = self.decide_delta_time(note_address, note_copy, preceding_address, preceding)
                start_time += delta
            note_copy.set_start_time(start_time)
            sequence.append(note_copy)
        if not sequence:
            return []
        first_note_time = sequence[0].start_time
        for note in sequence:
            note.start_time = note.start_time - first_note_time
        return sequence

    @staticmethod
    def get_pitch_string(note_sequence):
        return "".join([str(note.pitch) + " " for note in note_sequence])

    @staticmethod
    def decide_delta_time(note_to_add_address, note_to_add, current_address, current_note):
        if current_note is None:
            return 0
        return current_note.duration + current_note.next_start_delta

    def save_midi(self, sequence, output_file, tempo=120, sustain=False):
        ms = self.create_mido_sequence(sequence, tempo=tempo, sustain=sustain)
        ms.save(output_file)

    def create_mido_sequence(self, sequence, tempo=120, sustain=False):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        mido_sequence = []
        for note in sequence:
            try:
                mido_sequence.append(
                    mido.Message(
                        "note_on",
                        note=note.pitch,
                        velocity=note.velocity,
                        time=note.start_time,
                    )
                )
            except Exception:
                print("Something went wrong")
            mido_sequence.append(
                mido.Message(
                    "note_off",
                    note=note.pitch,
                    velocity=0,
                    time=note.start_time + note.duration,
                )
            )
        mido_sequence.sort(key=lambda messg: messg.time)
        if sustain:
            mido_sequence.insert(0, mido.Message("control_change", control=64, value=127, time=0))
        if tempo == -1 and self.tempo_msgs:
            average_tempo = int(np.sum(self.tempo_msgs) / len(self.tempo_msgs))
            mido_sequence.insert(0, mido.MetaMessage(type="set_tempo", tempo=average_tempo))
        current_time = 0
        for msg in mido_sequence:
            delta_in_beats = msg.time - current_time
            delta_in_ticks = int(mid.ticks_per_beat * delta_in_beats)
            msg.time = delta_in_ticks
            track.append(msg)
            current_time += delta_in_beats
        return mid

    def get_longest_subsequence_with_train(self, address_sequence):
        note_sequence = [self.get_input_note(address) for address in address_sequence]
        sequence_string = self.get_pitch_string(note_sequence)
        best = 0
        for input_seq in self._midi_store().input_sequences:
            train_string = self.get_pitch_string(input_seq)
            match = SequenceMatcher(None, train_string, sequence_string, autojunk=False).find_longest_match()
            nb_notes_common = train_string[match.a: match.a + match.size].count(" ")
            if nb_notes_common > best:
                best = nb_notes_common
        return best
