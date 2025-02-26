import os
from collections import Counter

import numpy as np
import mido
import random
import time
from difflib import SequenceMatcher

from mido import MetaMessage

"""
Implementation of Continuator different from original, to enable experiments with belief propagation and skips.
Build a representation for all contexts of size 1 to K and their continuations
Unlike in the original, Contexts AND continuations are viewpoints 
Realizations are kep separately and reused during sampling.
Sampling function is quite smart and attempts to avoid too long repetition (max order)
by avoiding singletons when it can
"""


class Note:
    def __init__(self, pitch, velocity, duration, start_time=0):
        self.pitch = pitch
        self.velocity = velocity
        # the duration in the original sequence
        self.duration = duration
        # the start time in the original sequence
        self.start_time = start_time
        self.overlaps_preceding =  False
        self.overlaps_next =  False

    def set_duration(self, d):
        self.duration = d

    def set_start_time(self, t):
        self.start_time = t

    def is_start_padding(self):
        return False

    def is_end_padding(self):
        return False

    def transpose(self, t):
        note =  Note(
            self.pitch + t, self.velocity, self.duration, start_time=self.start_time
        )
        note.overlaps_preceding = self.overlaps_preceding
        note.overlaps_next = self.overlaps_next
        return note

    def get_end_time(self):
        return self.start_time + self.duration

class Start_Padding(Note):
    def __init__(self):
        Note.__init__(self, -1, 0, 0)

    def is_start_padding(self):
        return True

    def transpose(self, t):
        return self


class End_Padding(Note):
    def __init__(self):
        Note.__init__(self, -2, 0, 0)

    def is_end_padding(self):
        return True

    def transpose(self, t):
        return self


class Continuator2:

    def __init__(self, midi_file, kmax=5, transposition=False):
        # the input sequences
        self.input_sequences = []
        self.tempo_msgs = []

        self.kmax = kmax
        # the list of realizations for a given viewpoint
        self.viewpoints_realizations = {}
        self.prefixes_to_continuations = np.empty(self.kmax, dtype=object)
        for k in range(self.kmax):
            self.prefixes_to_continuations[k] = {}

        # the original sequence
        notes_original = self.extract_notes(midi_file)
        # adds start and end notes
        notes_original = np.concatenate(([Start_Padding()], notes_original, [End_Padding()]))

        # learns, possibly in 12 transpositions
        trange = range(0, 1)
        if transposition:
            trange = range(-6, 6, 1)
        for t in trange:
            transposed = self.transpose_notes(notes_original, t)
            # store sequence in list of input sequences
            self.input_sequences.append(transposed)
            # learns one more sequence
            self.build_vo_markov_model(transposed)

    def get_input_note(self, note_address):
        # note_address is a tuple (melody index, index in melody)
        return self.input_sequences[note_address[0]][note_address[1]]

    def is_starting_address(self, note_address):
        return note_address[1] == 1

    def is_ending_address(self, note_address):
        return note_address[1] == len(self.input_sequences[note_address[0]]) - 2

    def transpose_notes(self, notes, t):
        return [n.transpose(t) for n in notes]

    def is_end_padding(self, cont_vp):
        return cont_vp == self.get_end_vp()

    def get_start_vp(self):
        return self.get_viewpoint(Start_Padding())

    def get_end_vp(self):
        return self.get_viewpoint(End_Padding())

    def random_initial_vp(self):
        # returns a random initial vp, which are continuations of start paddings
        return self.prefixes_to_continuations[0][tuple(self.get_start_vp())]

    def extract_notes(self, midi_file):
        """Extracts the sequence of note-on events from a MIDI file."""
        mid = mido.MidiFile(midi_file)
        notes = []
        pending_notes = np.empty(128, dtype=object)
        pending_start_times = np.zeros(128)
        current_time = 0
        for track in mid.tracks:
            for msg in track:
                current_time += msg.time
                if msg.type == 'set_tempo':
                    self.tempo_msgs.append(msg.tempo)
                if msg.type == "note_on" and msg.velocity > 0:
                    new_note = Note(msg.note, msg.velocity, 0)
                    notes.append(new_note)  # Store MIDI note number
                    pending_notes[msg.note] = new_note
                    pending_start_times[msg.note] = current_time
                    new_note.set_start_time(current_time)
                    new_note.set_duration(120)
                if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if pending_notes[msg.note] == None:
                        print("found 0 velocity note, skipping it")
                        continue
                    pending_note = pending_notes[msg.note]
                    duration = current_time - pending_start_times[msg.note]
                    pending_note.set_duration(duration)
                    pending_notes[msg.note] = None
                    pending_start_times[msg.note] = 0
        # sets the note status w/r their neighbors
        for i, note in enumerate(notes):
            if i > 0 :
                note.overlaps_preceding = note.start_time < notes[i-1].get_end_time()
            if i < len(notes) - 1 :
                note.overlaps_next = note.get_end_time() > notes[i+1].start_time
        return np.array(notes)

    def build_vo_markov_model(self, sequence):
        """Builds a variable-order Markov model for max K order
        accumulates with existing model"""
        # builds the vp sequence first
        vp_sequence = [self.get_viewpoint(note) for note in sequence]
        # add the realization to the viewpoint's realizations
        sequence_index = len(self.input_sequences) - 1
        for i, vp in enumerate(vp_sequence):
            if vp not in self.viewpoints_realizations:
                self.viewpoints_realizations[vp] = []
            self.viewpoints_realizations[vp].append(tuple([sequence_index, i]))
        # populate the prefixes_to_continuations with vp contexts to vps
        for k in range(self.kmax):
            prefixes_to_cont_k = self.prefixes_to_continuations[k]
            for i in range(len(vp_sequence) - k):
                if i < k + 1:
                    continue
                current_ctx = tuple(vp_sequence[i - k - 1 : i])
                if current_ctx not in prefixes_to_cont_k:
                    prefixes_to_cont_k[current_ctx] = []
                prefixes_to_cont_k[current_ctx].append(vp_sequence[i])
            self.prefixes_to_continuations[k] = prefixes_to_cont_k

    def get_first_order_matrix(self):
        # returns the matrix for first order Markov transitions
        # all states
        keys = sorted(self.prefixes_to_continuations[0])
        result = np.zeros((len(keys), len(keys)))
        for i_vp, vp in enumerate(keys):
            conts = self.prefixes_to_continuations[0][vp]
            occurrences = Counter(conts)
            for i_vp2, vp2 in enumerate(occurrences):
                result[i_vp, i_vp2] = occurrences[vp2]
            result[i_vp] /= result[i_vp].sum()
        return result

    def get_viewpoint(self, note):
        vp = tuple([note.pitch, (int)(note.duration / 10)])
        return vp

    def get_realizations_for_vp(self, vp):
        return self.viewpoints_realizations[vp]

    def random_starting_note(self):
        starting_vp = (-1, 0)
        starting_conts = self.get_realizations_for_vp(starting_vp)
        start = random.choice(starting_conts)
        return start

    def sample_sequence(self, start_vp, length=50):
        vp_seq = self.sample_vp_sequence(start_vp, length)
        seq = self.realize_vp_sequence(vp_seq)
        print (seq)
        return seq

    def sample_vp_sequence(self, start_vp, length=50):
        # Generates a new sequence of vps from the Markov model.
        current_seq = [start_vp]
        if length >= 0:
            # generate fixed length sequence
            for _ in range(length):
                cont = self.get_continuation(current_seq)
                if cont == -1:
                    print("restarting from scratch")
                    cont = self.random_initial_vp()
                current_seq.append(cont)
            return current_seq
        while True:
            cont = self.get_continuation(current_seq)
            if cont == -1:
                print("restarting from scratch")
                cont = self.random_initial_vp()
            current_seq.append(cont)
            if self.is_end_padding(cont):
                print("found the end")
                return current_seq
        return current_seq

    def get_continuation(self, current_seq):
        vp_to_skip = None
        for k in range(self.kmax, 0, -1):
            if k > len(current_seq):
                continue
            continuations_dict = self.prefixes_to_continuations[k - 1]
            viewpoint_ctx = tuple(current_seq[-k:])
            if viewpoint_ctx in continuations_dict:
                all_cont_vps = continuations_dict[viewpoint_ctx]
                # considers the number of different viewpoints, not the number of continuations as they are repeated
                if len(set(all_cont_vps)) == 1 and k > 1:
                    # proba to skip is proportional to order
                    if random.random() > (1 / (k + 1)):
                        # print(f"skipping continuation for {k=}")
                        vp_to_skip = all_cont_vps[0]
                        continue
                    else:
                        vp_to_skip = None
                        # print(f"not skipping singleton continuation for {k=}")
                if vp_to_skip is not None and k > 1:
                    conts_to_use = [c for c in all_cont_vps if c != vp_to_skip]
                else:
                    conts_to_use = all_cont_vps
                next_continuation = random.choice(conts_to_use)
                # print(
                #     f"found continuation for k {k} with cont size {len(all_cont_vps)}"
                # )
                return next_continuation
        print("no continuation found")
        return -1

    def realize_vp_sequence(self, vp_seq):
        result = []
        for i, vp in enumerate(vp_seq):
            if i == 1:
                initials = [real for real in self.viewpoints_realizations[vp] if self.is_starting_address(real)]
                result.append(random.choice(initials))
                continue
            if i == len(vp_seq) - 2 and vp_seq[-1] == End_Padding:
                lasts = [real for real in  self.viewpoints_realizations[vp]if self.is_ending_address(real)]
                result.append(random.choice(lasts))
                continue
            result.append(random.choice(self.viewpoints_realizations[vp]))
        return result


    def get_pitch_string(self, note_sequence):
        return "".join([str(note.pitch) + " " for note in note_sequence])

    def decide_delta_time(self, note_address, note, current_address, current_note):
        preceding_original_note = self.get_input_note(
            tuple([note_address[0], note_address[1] - 1]))
        interval_with_preceding_original = note.start_time - preceding_original_note.start_time
        interval_with_preceding_original_end = interval_with_preceding_original - preceding_original_note.duration
        overlap_with_original = note.start_time < (preceding_original_note.start_time + preceding_original_note.duration)

        current_next_original_note = None
        current_overlap_with_next_original =  False
        if current_address is not None:
            if not self.is_ending_address(current_address):
                current_next_original_note = self.get_input_note(tuple([current_address[0], current_address[1] + 1]))
                current_overlap_with_next_original = current_note.start_time + current_note.duration > current_next_original_note.start_time

        # if overlap_with_original and current_overlap_with_next_original:
        #     print("both overlap")
        # if not overlap_with_original and current_overlap_with_next_original:
        #     print("current overlap but not new note")
        # if  overlap_with_original and not current_overlap_with_next_original:
        #     print("current does not overlap but new note does")
        # if not overlap_with_original and not current_overlap_with_next_original:
        #     print("none overlap")

        # delta = interval_with_preceding_original_end + preceding_note.duration
        delta = interval_with_preceding_original
        delta = int(max(delta, 0))
        if current_note is not None and note.start_time + delta > current_note.get_end_time():
            print("create hole")
            delta = (int)(current_note.duration)
        # delta = (
        #     note.start_time
        #     - preceding_original_note.start_time
        # )
        return delta

    def save_midi(self, idx_sequence, output_file, tempo=120, sustain=False):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        # create a new sequence with the right start_times
        sequence = []
        start_time = 0
        for i, note_address in enumerate(idx_sequence):
            note = self.get_input_note(note_address)
            note_copy = Note(note.pitch, note.velocity, note.duration)
            # keeps the inter note time to be the same as in the original sequence
            if not self.is_starting_address(note_address):
                preceding = None
                preceding_address = None
                if len(sequence) > 1 :
                    preceding = sequence[-1]
                    preceding_address = idx_sequence[i - 1]
                delta = self.decide_delta_time(note_address, note, preceding_address, preceding)
                start_time += delta
            note_copy.set_start_time(start_time)
            sequence.append(note_copy)
        # shift the whole sequence to t=0
        first_note_time = sequence[0].start_time
        for note in sequence:
            note.start_time = note.start_time - first_note_time
        # create all mido messages and sort them
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
            except:
                print("Something went wrong")
            mido_sequence.append(
                mido.Message(
                    "note_off",
                    note=note.pitch,
                    velocity=0,
                    time=(note.start_time + (int)(note.duration)),
                )
            )
        mido_sequence.sort(key=lambda msg: msg.time)
        if sustain:
        # add pedal message
            mido_sequence.insert(0, mido.Message(
                        "control_change",
                        control=64,
                        value=127,
                        time=0,
                    ))
        if tempo == -1:
            # takes the origina average tempo
            average_tempo = (int) (np.sum(self.tempo_msgs) / len(self.tempo_msgs))
            mido_sequence.insert(0, mido.MetaMessage(type='set_tempo', tempo=average_tempo))
        current_time = 0
        for msg in mido_sequence:
            delta = msg.time - current_time
            msg.time = delta
            track.append(msg)
            current_time += delta
        mid.save(output_file)
        # plays the file approximately, can be heard of Logic is open
        # with mido.open_output() as output:
        #     for note in sequence:
        #         output.send(mido.Message('note_on', note=note, velocity=64))
        #         time.sleep(0.2)
        #         output.send(mido.Message('note_off', note=note, velocity=64))

    def get_longest_subsequence_with_train(self, address_sequence):
        note_sequence = [self.get_input_note(address) for address in address_sequence]
        sequence_string = generator.get_pitch_string(note_sequence)
        best = 0
        for input_seq in self.input_sequences:
            train_string = generator.get_pitch_string(input_seq)
            match = SequenceMatcher(
                None, train_string, sequence_string, autojunk=False
            ).find_longest_match()
            nb_notes_common = train_string[match.a : match.a + match.size].count(" ")
            if nb_notes_common > best:
                best = nb_notes_common
        return best

    def show_conts_structure(self):
        for k in range(generator.kmax):
            print(
                f"size of contexts of size {k}: {len(generator.prefixes_to_continuations[k])}"
            )
        # looks at the sparsity of the matrix
        order1 = generator.prefixes_to_continuations[0]
        voc_size = len(order1)
        min = voc_size
        max = 0
        for voc in order1.keys():
            conts_size = len(set(order1[voc]))
            if conts_size > max:
                max = conts_size
            if conts_size < min:
                min = conts_size
        print(f"voc size: {voc_size}")
        print(f"min order 1 size: {min}, max: {max}")


# midi_file_path = "../../data/Ravel_jeaux_deau.mid"
# midi_file_path = "../../data/debussy_prelude_short.mid"
# midi_file_path = "../../data/prelude_c_expressive.mid"
# midi_file_path = "../../data/prelude_c.mid"
# midi_file_path = "../../data/bach_partita_mono.midi"
midi_file_path = "../../data/keith/K1.mid"
t0 = time.perf_counter_ns()
generator = Continuator2(midi_file_path, 5, transposition=True)
# matrix = generator.get_first_order_matrix()
# print(matrix.shape)
# t1 = time.perf_counter_ns()
# print(f"total time: {(t1 - t0) / 1000000}")
# Sampling a new sequence from the  model
generated_sequence = generator.sample_sequence(generator.get_start_vp(), length=-1)
# print(f"generated sequence of length {len(generated_sequence)}")
generator.save_midi(generated_sequence[1:-1], "../../data/ctor2_keith_K1.mid", tempo= -1, sustain=True)
# os.system("say sequence generated &")
# print("Generated Sequence:", generated_sequence)
# print("computing plaggiarism:")
# print(
#     f"{generator.get_longest_subsequence_with_train(generated_sequence)} successive notes in commun with train"
# )
generator.show_conts_structure()

