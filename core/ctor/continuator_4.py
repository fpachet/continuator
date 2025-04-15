import pathlib
from collections import Counter

import numpy as np
import mido
import random
import time
from difflib import SequenceMatcher

from variable_order_markov import Variable_order_Markov

"""
- Split the music Continuator class from a generic Variable Order Markov, usable for any type of sequence (e.g. words).
- Implementation of Continuator different from original, to enable experiments with belief propagation and skips.
- Representation of contexts of size 1 to K and their continuations with dictionaries. Trees/oracles are useless here.
- Contexts are tuples of viewpoints AND continuations are viewpoints (see get_viewpoint()) (Unlike in the original)
- Realizations are kept separately for each vp and reused during sampling. They are represented as addresses, i.e. tuple (index_of_melody, index_in_melody)
- Sampling attempts to avoid too long repetitions (a kind of max-order) by avoiding singletons when it can
- Sampling is performed both by belief propagation (1st order) and by variable-order and combined
- Realization of viewpoints is performed with dynamic programming, à la HMM
- Representation of polyphony is different from original Continuator. Clusters are not considered, only notes.
They have a "status" describing how they were played originally, which is preserved at sampling. This enables more creativity for chords.
- TODO: add real-time input
- TODO: audio synthesis with Dawdreamer
- TODO: add database storage of real time performances
- TODO: pre-train on large corpus of melodies
- TODO: data augmentation with inversions, negative harmony, etc.
- TODO: rhythm transfer for data augmentation/control
- TODO: server with js client, or huggingface solution or github page with python2js
- TODO: use fine-tuning of transformers
"""


class Note:
    def __init__(self, pitch, velocity, duration, start_time=0):
        self.pitch = pitch
        self.velocity = velocity
        # the duration in the original sequence in beats, assuming 120bpm
        self.duration = duration
        # the start time in the original sequence in beats, assuming 120bpm
        self.start_time = start_time
        # time between start and the start of preceding note, always > 0
        self.preceding_start_delta = 0 # in beats, assuming 120bpm
        # time between start and the end of preceding note. Negative if overlaps with preceding
        self.preceding_end_delta = 0 # in beats, assuming 120bpm
        # time between start of next note and end. Negative if overlaps with next
        self.next_start_delta = 0 # in beats, assuming 120bpm
        # time between end of next note and end
        self.next_end_delta = 0 # in beats, assuming 120bpm

    def __str__(self):
        return f"{self.pitch} @ [{self.start_time}, {self.get_end_time()}]"

    def __repr__(self):
        return f"{self.pitch} @ [{self.start_time}, {self.get_end_time()}]"

    def set_duration(self, d):
        self.duration = d

    def set_start_time(self, t):
        self.start_time = t

    def overlaps_left(self):
        # if overlap is greater than half the duration
        return self.preceding_end_delta < 0

    def overlaps_right(self):
        # if overlap is greater than half the duration
        return self.next_start_delta < 0

    def transpose(self, t):
        note = self.copy()
        note.pitch = self.pitch + t
        return note

    def copy(self):
        new_note = Note(self.pitch, self.velocity, self.duration, start_time=self.start_time)
        new_note.preceding_start_delta = self.preceding_start_delta
        new_note.preceding_end_delta = self.preceding_end_delta
        new_note.next_start_delta = self.next_start_delta
        new_note.next_end_delta = self.next_end_delta
        return new_note

    def get_end_time(self):
        return self.start_time + self.duration

    def is_compatible_with(self, note):
        # returns true if self and note have same polyphonic status
        return self.overlaps_right() == note.overlaps_left()

    def get_status_right(self):
        if self.next_end_delta <= 0:
            return 'inside'
        if self.next_start_delta < 0:
            return 'overlaps'
        return 'after'

    def get_status_left(self):
        if self.preceding_end_delta >= 0:
            return 'before'
        if abs(self.preceding_end_delta) < self.duration:
            return 'overlaps'
        return 'contains'

    def is_similar_realization(self, note):
        if self.pitch != note.pitch:
            return False
        if self.velocity != note.velocity:
            return False
        if self.duration != note.duration:
            return False
        if self.preceding_end_delta != note.preceding_end_delta:
            return False
        if self.preceding_start_delta != note.preceding_start_delta:
            return False
        if self.next_start_delta != note.next_start_delta:
            return False
        if self.next_end_delta != note.next_end_delta:
            return False
        return True

class Continuator2:

    def __init__(self, midi_file=None, kmax=5, transposition=False):
        self.vom = Variable_order_Markov(None, self.get_viewpoint, kmax)
        self.tempo_msgs = []
        if midi_file is not None:
            self.learn_file(midi_file, transposition)

    @staticmethod
    def get_viewpoint(note):
        vp = tuple([note.pitch, int(note.duration / 5), note.overlaps_left(), note.overlaps_right()])
        # vp = tuple([note.pitch, (int)(note.duration / 10)])
        return vp

    def get_phrase_titles(self):
        cpt = 1
        result = []
        for phrase in self.vom.input_sequences:
            result.append(str(cpt) + " phrase " + str(len(phrase)))
            cpt+=1
        return result

    def get_phrase(self, index):
        return self.vom.input_sequences[index]

    def learn_file(self, midi_file, transposition):
        notes_original = self.extract_notes(midi_file)
        self.learn_phrase(notes_original, transposition)

    def learn_phrase(self, note_sequence, transposition):
        all_pitches = [note.pitch for note in note_sequence]
        print(f"number of different pitches in train: {len(Counter(all_pitches))}")
        print(f"min pitch: {min(all_pitches)}, max pitch: {max(all_pitches)}")
        # learns, possibly in 12 transpositions
        trange = range(0, 1)
        if transposition:
            trange = range(-6, 6, 1)
        for t in trange:
            transposed = self.transpose_notes(note_sequence, t)
            # learns one more sequence
            self.vom.learn_sequence(transposed)

    def learn_files(self, files, transposition=False):
        # suppose at least one file has been learned already
        for file in files:
            self.learn_file(file, transposition)

# mido gives time in milliseconds from real input. Converts it into beast, assuming 120bpm
    def learn_phrase_from_mido(self, phrase):
        self.learn_phrase(self.get_phrase_from_mido(phrase), False)

    def get_phrase_from_mido(self, phrase):
        sequence = []
        pending_notes = {}
# assign ABSOLUTE TIME to each message first, by cumulating all the deltas
# time here is in milliseconds
        start_time = 0
        for msg in phrase:
            start_time = start_time + msg.time
            msg.time = start_time
# joins note on and note off
        for msg in phrase:
            if msg.type == "note_on":
                 pending_notes[msg.note] = msg
            else:
                if msg.type == "note_off":
                    note_on_msg = pending_notes[msg.note]
                    if note_on_msg is None:
                        print('problem')
                    else:
                        start_time = note_on_msg.time * 2 #seconds to beat at 120 bpm
                        duration = (msg.time - note_on_msg.time) * 2
                        new_note = Note(note_on_msg.note, note_on_msg.velocity, duration, start_time)
                        sequence.append(new_note)
        self.set_delta_notes(sequence)
        return sequence

    def transpose_notes(self, notes, t):
        return [n.transpose(t) for n in notes]

    def get_input_note(self, note_address):
        # note_address is a tuple (melody index, index in melody)
        return self.vom.get_input_object(note_address)

    def is_starting_address(self, note_address):
        return self.vom.is_starting_address(note_address)

    def is_ending_address(self, note_address):
        return self.vom.is_ending_address(note_address)

    def get_start_vp(self):
        return self.vom.start_padding

    def get_end_vp(self):
        return self.vom.end_padding

# @ time in midifile is expressed in ticks with some resolution. We convert it into beats, assuming 120bpm
    def extract_notes(self, midi_file):
        """Extracts the sequence of note-on events from a MIDI file."""
        mid = mido.MidiFile(midi_file)
        resolution = mid.ticks_per_beat
        notes = []
        pending_notes = np.empty(128, dtype=object)
        pending_start_times = np.zeros(128)
        current_time = 0
        for track in mid.tracks:
            for msg in track:
                current_time += 2 * mido.tick2second(msg.time, ticks_per_beat=resolution,tempo= 500000) # in beats
                if msg.type == 'set_tempo':
                    self.tempo_msgs.append(msg.tempo)
                if msg.type == "note_on" and msg.velocity > 0:
                    new_note = Note(msg.note, msg.velocity, 0)
                    notes.append(new_note)  # Store MIDI note number
                    pending_notes[msg.note] = new_note
                    pending_start_times[msg.note] = current_time
                    new_note.set_start_time(current_time)
                    new_note.set_duration(1) # beat
                if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if pending_notes[msg.note] is None:
                        print("found 0 velocity note, skipping it")
                        continue
                    pending_note = pending_notes[msg.note]
                    duration = current_time - pending_start_times[msg.note]
                    pending_note.set_duration(duration)
                    pending_notes[msg.note] = None
                    pending_start_times[msg.note] = 0
        # sets the note status w/r their neighbors
        self.set_delta_notes(notes)
        return np.array(notes)

    def set_delta_notes(self, notes):
        for i, note in enumerate(notes):
            if i > 0:
                note.preceding_start_delta = note.start_time - notes[i - 1].start_time
                note.preceding_end_delta = note.start_time - notes[i - 1].get_end_time()
            if i < len(notes) - 1:
                note.next_start_delta = notes[i + 1].start_time - note.get_end_time()
                note.next_end_delta = notes[i + 1].get_end_time() - note.get_end_time()

    @staticmethod
    def all_midi_files_from_path(path_string):
        path = pathlib.Path(path_string)
        return list(path.glob('*.mid')) + list(path.glob('*.midi'))

    def sample_sequence(self, length=50, constraints=None):
        """
        :type constraints: dict
        """
        return self.vom.sample_sequence(length, constraints=constraints)

    def realize_vp_sequence(self, vp_seq):
        print(f"realize sequence of {len(vp_seq)} viewpoints")
        note_sequence = []
        for i, vp in enumerate(vp_seq):
            if i == 0:
                initials = [real for real in self.vom.viewpoints_realizations[vp] if self.is_starting_address(real)]
                if len(initials) != 0:
                    note_sequence.append(random.choice(initials))
                    continue
            if i == len(vp_seq) - 1 and vp_seq[-1] == self.vom.end_padding:
                lasts = [real for real in self.vom.viewpoints_realizations[vp] if self.is_ending_address(real)]
                if len(lasts) != 0:
                    note_sequence.append(random.choice(lasts))
                    continue
            note_sequence.append(random.choice(self.vom.viewpoints_realizations[vp]))

        # domains = [self.viewpoints_realizations[vp] for vp in vp_seq]
        # # # try to put together notes with compatible status @TODO
        # unary_cost = lambda i, real: 0
        # binary_cost = lambda i, real1, j, real2: (int)(not self.get_input_note(real1).is_compatible_with(self.get_input_note(real2)))
        # optimizer = VariableDomainSequenceOptimizer(domains, unary_cost, binary_cost)
        # cost, best_seq = optimizer.fit()

        result = self.set_timing(note_sequence)
        return result
        # return best_seq

    def get_vp_for_pitch(self, pitch):
        # this is way too costly, but used only at constraint initialization. Can be cached
        vps = []
        for vp, notes in self.vom.viewpoints_realizations.items():
            for note_address in notes:
                note = self.vom.get_input_object(note_address)
                if note.pitch == pitch:
                    vps.append(vp)
        return random.choice(vps)

    def set_timing(self, idx_sequence):
        sequence = []
        start_time = 0
        for i, note_address in enumerate(idx_sequence):
            note_copy = self.get_input_note(note_address).copy()
            # keeps the inter note time to be the same as in the original sequence
            if len(sequence) > 0:
                preceding = sequence[-1]
                preceding_address = idx_sequence[i - 1]
                delta = self.decide_delta_time(note_address, note_copy, preceding_address, preceding)
                start_time += delta
            note_copy.set_start_time(start_time)
            sequence.append(note_copy)
        # shift the whole sequence to t=0
        first_note_time = sequence[0].start_time
        for note in sequence:
            note.start_time = note.start_time - first_note_time
        return sequence

    def get_pitch_string(self, note_sequence):
        return "".join([str(note.pitch) + " " for note in note_sequence])

    def decide_delta_time(self, note_to_add_address, note_to_add, current_address, current_note):
        if current_note is None:
            return 0
        cur_status = current_note.get_status_right()
        note_to_add_status = note_to_add.get_status_left()
        delta = current_note.duration + current_note.next_start_delta
        # print(cur_status + '  ' + note_to_add_status)
        if cur_status == "inside":
            if note_to_add_status == "before":
                return delta
            if note_to_add_status == "overlaps":
                return delta
            if note_to_add_status == "contains":
                return delta
        if cur_status == "overlaps":
            if note_to_add_status == "before":
                return delta
            if note_to_add_status == "overlaps":
                return delta
            if note_to_add_status == "contains":
                return delta
        if cur_status == "after":
            if note_to_add_status == "before":
                return delta
            if note_to_add_status == "overlaps":
                return delta
            if note_to_add_status == "contains":
                return delta
        print("should not be here")
        return 0

    def save_midi(self, sequence, output_file, tempo=120, sustain=False):
        ms = self.create_mido_sequence(sequence, tempo=tempo, sustain=False)
        ms.save(output_file)

    def create_mido_sequence(self, sequence, tempo=120, sustain=False):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        # create a new sequence with the right start_times
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
                    time=note.start_time + note.duration,
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
        if tempo == -1 and len(self.tempo_msgs) > 0:
            # takes the original average tempo
            average_tempo = (int)(np.sum(self.tempo_msgs) / len(self.tempo_msgs))
            mido_sequence.insert(0, mido.MetaMessage(type='set_tempo', tempo=average_tempo))
        current_time = 0
        # converts beats into ticks, assuming 480 ticks per second
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
        for input_seq in self.vom.input_sequences:
            train_string = self.get_pitch_string(input_seq)
            match = SequenceMatcher(
                None, train_string, sequence_string, autojunk=False
            ).find_longest_match()
            nb_notes_common = train_string[match.a: match.a + match.size].count(" ")
            if nb_notes_common > best:
                best = nb_notes_common
        return best


if __name__ == '__main__':
    # midi_file_path = "../../data/Ravel_jeaux_deau.mid"
    # midi_file_path = "../../data/test_sequence_3notes.mid"
    # midi_file_path = "../../data/test_sequence_arpeggios.mid"
    # midi_file_path = "../../data/debussy_prelude.mid"
    # midi_file_path = "../../data/prelude_c_expressive.mid"
    # midi_file_path = "../../data/prelude_c_linear.mid"
    # midi_file_path = "../../data/partita_piano_1/pr1_1_joined.mid"
    # midi_file_path = "../../data/take6/A_quiet_place_joined.mid"
    # midi_file_path = "../../data/prelude_c_expressive.mid"
    midi_file_path = "../../data/prelude_c.mid"
    # midi_file_path = "../../data/bach_partita_mono.midi"
    # midi_file_path = "../../data/keith/train/K7_MD.mid"
    # midi_file_path = "../../../maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_03_Track03_wav--1.midi"
    t0 = time.perf_counter_ns()
    generator = Continuator2(midi_file_path, 4, transposition=False)
    # matrix = generator.get_first_order_matrix()
    # print(matrix.shape)
    # t1 = time.perf_counter_ns()
    # print(f"total time: {(t1 - t0) / 1000000}")
    # Sampling a new sequence from the  model
    constraints = {}
    # constraints[0] = generator.get_start_vp()
    constraints[0] = generator.get_vp_for_pitch(62)
    constraints[19] = generator.get_end_vp()
    generated_sequence = generator.sample_sequence(length=20, constraints=constraints)
    t1 = time.perf_counter_ns()
    print(f"total time: {(t1 - t0) / 1_000_000}ms")
    # print(f"generated sequence of length {len(generated_sequence)}")
    sequence_to_render = generated_sequence[0:-1]
    rendered_sequence = generator.realize_vp_sequence(sequence_to_render)
    generator.save_midi(rendered_sequence, "../../data/ctor2_output.mid", tempo=-1, sustain=False)
    # pmpr = generator.create_pr®etty_midi_pr(generated_sequence)
    # generator.plot_piano_roll(pmpr)
    # os.system("say sequence generated &")
    # print("Generated Sequence:", generated_sequence)
    # print("computing plagiarism:")
    # print(
    #     f"{generator.get_longest_subsequence_with_train(generated_sequence)} successive notes in commun with train"
    # )
    generator.vom.show_conts_structure()
