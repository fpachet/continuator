import pathlib
from collections import Counter

import numpy as np
import mido
import random
import time
from difflib import SequenceMatcher
import pretty_midi
import matplotlib.pyplot as plt

from core.ctor.belief_propag_stringham_clean import PGM, LabeledArray, Messages
from core.ctor.dynaprog import VariableDomainSequenceOptimizer

"""
- Implementation of Continuator different from original, to enable experiments with belief propagation and skips.
- Representation of contexts of size 1 to K and their continuations with dictionaries. Trees/oracles are useless here.
- Contexts are tuples of viewpoints AND continuations are viewpoints (see get_viewpoint()) (Unlike in the original)
- Realizations are kept separately for each vp and reused during sampling. They are represented as addresses, i.e. tuple (index_of_melody, index_in_melody)
- Sampling attempts to avoid too long repetitions (a kind of max-order) by avoiding singletons when it can
- Sampling is performed both by belief propagation (1st order) and by variable-order and combined
- Realization of viewpoints is performed with dynamic programming, à la HMM
- Representation of polyphony is different from original Continuator. Clusters are not considered, only notes.
They have a "status" describing how they were played originally, which is preserved at sampling. This enables more creativity for chords.
- TODO: finish sampling with BP
- TODO: finish handling polyphony and note status
- TODO: audio synthesis with Dawdreamer
- TODO: add real-time input
- TODO: add database storage of real time performances
- TODO: pre-train on large corpus of melodies
- TODO: data augmentation with inversions, negative harmony, etc.
- TODO: rhythm transfer for data augmentation/control
- TODO: server with js client, or hugginface solution or github page with python2js
- TODO: use continuator when possible
- TODO: use fine-tuning of transformers
"""


class Note:
    def __init__(self, pitch, velocity, duration, start_time=0):
        self.pitch = pitch
        self.velocity = velocity
        # the duration in the original sequence
        self.duration = duration
        # the start time in the original sequence
        self.start_time = start_time
        # time between start and the start of preceding note, always > 0
        self.preceding_start_delta = 0
        # time between start and the end of preceding note. Negative if overlaps with preceding
        self.preceding_end_delta = 0
        # time between start of next note and end. Negative if overlaps with next
        self.next_start_delta = 0
        # time between end of next note and end
        self.next_end_delta = 0

    def set_duration(self, d):
        self.duration = d

    def set_start_time(self, t):
        self.start_time = t

    def is_start_padding(self):
        return False

    def is_end_padding(self):
        return False

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
        new_note = Note(
            self.pitch, self.velocity, self.duration, start_time=self.start_time
        )
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
            return "inside"
        if self.next_start_delta < 0:
            return "overlaps"
        return "after"

    def get_status_left(self):
        if self.preceding_end_delta >= 0:
            return "before"
        if abs(self.preceding_end_delta) < self.duration:
            return "overlaps"
        return "contains"

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
        # needs a fixed list of viewpoint to build the markov matrix
        self.all_unique_viewpoints = []

        self.kmax = kmax
        # the list of realizations for a given viewpoint
        self.viewpoints_realizations = {}
        self.prefixes_to_continuations = np.empty(self.kmax, dtype=object)
        for k in range(self.kmax):
            self.prefixes_to_continuations[k] = {}

        self.learn_file(midi_file, transposition)

    def learn_file(self, midi_file, transposition):
        # the original sequence
        notes_original = self.extract_notes(midi_file)

        all_pitches = [note.pitch for note in notes_original]
        print(f"number of different pitches in train: {len(Counter(all_pitches))}")
        print(f"min pitch: {min(all_pitches)}, max pitch: {max(all_pitches)}")
        # adds start and end notes
        notes_original = np.concatenate(
            ([Start_Padding()], notes_original, [End_Padding()])
        )
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

    def learn_files(self, files, transposition=False):
        # suppose at least one file has been learned already
        for file in files:
            self.learn_file(file, transposition)

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

    def voc_size(self):
        # the number of unique viewpoints, including Start and End viewpoints
        # StartPadding and endPadding are included
        return len(self.all_unique_viewpoints)

    def random_initial_vp(self):
        # returns a random initial vp, which are continuations of start paddings
        return self.prefixes_to_continuations[0][tuple(self.get_start_vp())]

    def get_all_unique_viewpoints(self):
        return self.all_unique_viewpoints

    def index_of_vp(self, vp):
        return self.all_unique_viewpoints.index(vp)

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
                if msg.type == "set_tempo":
                    self.tempo_msgs.append(msg.tempo)
                if msg.type == "note_on" and msg.velocity > 0:
                    new_note = Note(msg.note, msg.velocity, 0)
                    notes.append(new_note)  # Store MIDI note number
                    pending_notes[msg.note] = new_note
                    pending_start_times[msg.note] = current_time
                    new_note.set_start_time(current_time)
                    new_note.set_duration(120)
                if msg.type == "note_off" or (
                        msg.type == "note_on" and msg.velocity == 0
                ):
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
            if i > 0:
                note.preceding_start_delta = note.start_time - notes[i - 1].start_time
                note.preceding_end_delta = note.start_time - notes[i - 1].get_end_time()
            if i < len(notes) - 1:
                note.next_start_delta = notes[i + 1].start_time - note.get_end_time()
                note.next_end_delta = notes[i + 1].get_end_time() - note.get_end_time()
        return np.array(notes)

    def all_midi_files_from_path(self, path_string):
        path = pathlib.Path(path_string)
        return list(path.glob("*.mid")) + list(path.glob("*.midi"))

    def build_vo_markov_model(self, sequence):
        """Builds a variable-order Markov model for max K order
        accumulates with existing model"""
        # builds the vp sequence first
        vp_sequence = [self.get_viewpoint(note) for note in sequence]
        # adds unique viewpoints if any
        for vp in vp_sequence:
            if vp not in self.all_unique_viewpoints:
                self.all_unique_viewpoints.append(vp)
        # add the realization to the viewpoint's realizations
        sequence_index = len(self.input_sequences) - 1
        for i, vp in enumerate(vp_sequence):
            if vp not in self.viewpoints_realizations:
                self.viewpoints_realizations[vp] = []
            self.add_viewpoint_realization(i, sequence_index, vp)
        # populate the prefixes_to_continuations with vp contexts to vps
        for k in range(self.kmax):
            prefixes_to_cont_k = self.prefixes_to_continuations[k]
            for i in range(len(vp_sequence) - k):
                if i < k + 1:
                    continue
                current_ctx = tuple(vp_sequence[i - k - 1: i])
                if current_ctx not in prefixes_to_cont_k:
                    prefixes_to_cont_k[current_ctx] = []
                prefixes_to_cont_k[current_ctx].append(vp_sequence[i])
            self.prefixes_to_continuations[k] = prefixes_to_cont_k
        # special case for the endVp, which has no continuation, but should be in the list for consistency
        end_tuple = tuple(self.get_end_vp())
        if end_tuple not in self.prefixes_to_continuations[0]:
            # ends goes to end
            self.prefixes_to_continuations[0][tuple([end_tuple])] = [self.get_end_vp()]

    def add_viewpoint_realization_old(self, i, sequence_index, vp):
        new_address = tuple([sequence_index, i])
        self.viewpoints_realizations[vp].append(new_address)

    def add_viewpoint_realization_new(self, i, sequence_index, vp):
        # adds only if different from existing ones, to avoid inflation in case of monotonous pieces
        new_address = tuple([sequence_index, i])
        if self.is_starting_address(new_address) or self.is_ending_address(new_address):
            # starting address are added, cause useful at rendering time
            self.viewpoints_realizations[vp].append(new_address)
            return
        new_note = self.get_input_note(new_address)
        for real in self.viewpoints_realizations[vp]:
            real_note = self.get_input_note(real)
            if real_note.is_similar_realization(new_note):
                return
        self.viewpoints_realizations[vp].append(new_address)

    add_viewpoint_realization = add_viewpoint_realization_new

    def get_priors(self):
        key_counts = {
            key: len(continuations)
            for key, continuations in self.viewpoints_realizations.items()
        }
        total_count = sum(key_counts.values())
        priors = {key: count / total_count for key, count in key_counts.items()}
        # Step 4: Convert to a sorted vector (optional)
        sorted_keys = self.get_all_unique_viewpoints()  # Ensure consistent ordering
        probability_vector = np.array([priors[key] for key in sorted_keys])
        return probability_vector

    def sample_zero_order(self, k):
        priors = self.get_priors()
        vp_seq = random.choices(self.get_all_unique_viewpoints(), weights=priors, k=k)
        return self.realize_vp_sequence(vp_seq)

    def get_first_order_matrix(self):
        # returns the matrix for first order Markov transitions
        # all states. This includes start and end padding states
        keys = self.get_all_unique_viewpoints()
        result = np.zeros((len(keys), len(keys)))
        k0 = self.prefixes_to_continuations[0]
        for i_vp, vp in enumerate(keys):
            conts = k0[tuple([vp])]
            occurrences = Counter(conts)
            for vp2 in occurrences:
                i_vp2 = keys.index(vp2)
                result[i_vp, i_vp2] = occurrences[vp2]
            result[i_vp] /= result[i_vp].sum()
        return result

    def get_viewpoint(self, note):
        vp = tuple(
            [
                note.pitch,
                (int)(note.duration / 5),
                note.overlaps_left(),
                note.overlaps_right(),
            ]
        )
        # vp = tuple([note.pitch, (int)(note.duration / 10)])
        return vp

    def get_realizations_for_vp(self, vp):
        return self.viewpoints_realizations[vp]

    def random_starting_note(self):
        starting_vp = (-1, 0)
        starting_conts = self.get_realizations_for_vp(starting_vp)
        start = random.choice(starting_conts)
        return start

    def sample_sequence(self, start_vp, length=50):
        # pgm = self.build_bp_graph(start_vp, length, self.get_end_vp())
        # sets constraints on start and end
        # pgm.set_value('x1', self.index_of_vp(start_vp))
        # pgm.set_value('x' + str(length + 2), self.index_of_vp(self.get_end_vp()))

        # without BP
        vp_seq = self.sample_vp_sequence(start_vp, length)
        # with BP
        # vp_seq = self.sample_vp_sequence_with_bp(start_vp, length, pgm)
        seq = self.realize_vp_sequence(vp_seq)
        return seq

    def build_bp_graph(self, start_vp, length, end_vp):
        string = ""
        # length of bp graph is length + 2: plus the start (possibly the end of an existing sequence) and plus the end viewpoint
        seq_length = length + 2
        for i in range(1, seq_length + 1):
            string = string + "p(x" + str(i) + ")"
        for i in range(2, seq_length + 1):
            string = string + "p(x" + str(i) + "|x" + str(i - 1) + ")"
        pgm = PGM.from_string(string)
        mat = LabeledArray(
            np.array(self.get_first_order_matrix()).transpose(),
            ["x2", "x1"],
        )
        # assert is_conditional_prob(mat, "x2")
        m = self.voc_size()
        data_dict = {}
        for i in range(seq_length):
            variable_dist = np.random.uniform(1 / m, 1 / m, m)
            # should avoid start and end values
            variable_dist[self.index_of_vp(start_vp)] = 0
            variable_dist[self.index_of_vp(end_vp)] = 0
            variable_dist /= variable_dist.sum()
            data_dict["p(x" + str(i + 1) + ")"] = LabeledArray(
                np.array(variable_dist), ["x" + str(i + 1)]
            )
            data_dict["p(x" + str(i + 2) + "|x" + str(i + 1) + ")"] = LabeledArray(
                mat.array, ["x" + str(i + 2), "x" + str(i + 1)]
            )
        pgm.set_data(data_dict)
        return pgm

    def sample_vp_sequence_with_bp(self, start_vp, length, pgm):
        # Generates a new sequence of vps from the Markov model.
        if length < 0:
            print("impossible")
        current_seq = [start_vp]
        # generate fixed length sequence
        pgm.print_marginals()
        for i in range(length):
            marginal_i = Messages().marginal(pgm.variable_from_name("x" + str(i + 2)))
            # compare with the markov transition matrix
            self.get_first_order_matrix()[self.index_of_vp(self.get_start_vp())]
            cont = self.get_continuation(current_seq)
            if cont == -1:
                print("restarting from scratch")
                cont = self.random_initial_vp()
            current_seq.append(cont)
        return current_seq

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
        print(f"realize sequence of {len(vp_seq)} viewpoints")
        note_sequence = []
        for i, vp in enumerate(vp_seq):
            if i == 1:
                initials = [
                    real
                    for real in self.viewpoints_realizations[vp]
                    if self.is_starting_address(real)
                ]
                if len(initials) != 0:
                    note_sequence.append(random.choice(initials))
                    continue
            if i == len(vp_seq) - 2 and vp_seq[-1] == End_Padding:
                lasts = [
                    real
                    for real in self.viewpoints_realizations[vp]
                    if self.is_ending_address(real)
                ]
                if len(lasts) != 0:
                    note_sequence.append(random.choice(lasts))
                    continue
            note_sequence.append(random.choice(self.viewpoints_realizations[vp]))

        # domains = [self.viewpoints_realizations[vp] for vp in vp_seq]
        # # # try to put together notes with compatible status @TODO
        # unary_cost = lambda i, real: 0
        # binary_cost = lambda i, real1, j, real2: (int)(not self.get_input_note(real1).is_compatible_with(self.get_input_note(real2)))
        # optimizer = VariableDomainSequenceOptimizer(domains, unary_cost, binary_cost)
        # cost, best_seq = optimizer.fit()

        result = self.set_timing(note_sequence[1:-1])
        return result
        # return best_seq

    def set_timing(self, idx_sequence):
        sequence = []
        start_time = 0
        for i, note_address in enumerate(idx_sequence):
            note_copy = self.get_input_note(note_address).copy()
            # keeps the inter note time to be the same as in the original sequence
            if len(sequence) > 0:
                preceding = sequence[-1]
                preceding_address = idx_sequence[i - 1]
                delta = (int)(
                    self.decide_delta_time(
                        note_address, note_copy, preceding_address, preceding
                    )
                )
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

    def decide_delta_time_old(self, note_address, note, current_address, current_note):
        preceding_original_note = self.get_input_note(
            tuple([note_address[0], note_address[1] - 1])
        )
        interval_with_preceding_original = (
                note.start_time - preceding_original_note.start_time
        )

        if current_address is not None:
            if not self.is_ending_address(current_address):
                current_next_original_note = self.get_input_note(
                    tuple([current_address[0], current_address[1] + 1])
                )
                current_overlap_with_next_original = (
                        current_note.start_time + current_note.duration
                        > current_next_original_note.start_time
                )

        # delta = interval_with_preceding_original_end + preceding_note.duration
        delta = interval_with_preceding_original
        delta = int(max(delta, 0))
        if (
                current_note is not None
                and note.start_time + delta > current_note.get_end_time()
        ):
            print("create hole")
            delta = (int)(current_note.duration)
        return delta

    def decide_delta_time(
            self, note_to_add_address, note_to_add, current_address, current_note
    ):
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

    def create_pretty_midi_pr(self, note_sequence):
        # For each note in the list, create a pretty_midi.Note object and add to the instrument
        pm = pretty_midi.PrettyMIDI()
        # Create an Instrument instance for a specific program (sound)
        instrument = pretty_midi.Instrument(program=0)
        for note in note_sequence:
            note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start_time,
                end=note.get_end_time(),
            )
            instrument.notes.append(note)
        # Add the instrument to the PrettyMIDI object
        pm.instruments.append(instrument)
        return pm.get_piano_roll(fs=100)

    def plot_piano_roll(piano_roll, fs=100):
        # Plot using matplotlib
        plt.figure()
        plt.imshow(piano_roll, aspect="auto", origin="lower", interpolation="nearest")
        plt.xlabel("Time (frames at fs = {})".format(fs))
        plt.ylabel("MIDI pitch")
        plt.title("PrettyMIDI Piano Roll")
        plt.show()

    def save_midi(self, sequence, output_file, tempo=120, sustain=False):
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
                    time=(note.start_time + (int)(note.duration)),
                )
            )
        mido_sequence.sort(key=lambda msg: msg.time)
        if sustain:
            # add pedal message
            mido_sequence.insert(
                0,
                mido.Message(
                    "control_change",
                    control=64,
                    value=127,
                    time=0,
                ),
            )
        if tempo == -1 and len(self.tempo_msgs) > 0:
            # takes the original average tempo
            average_tempo = (int)(np.sum(self.tempo_msgs) / len(self.tempo_msgs))
            mido_sequence.insert(
                0, mido.MetaMessage(type="set_tempo", tempo=average_tempo)
            )
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
        sequence_string = self.get_pitch_string(note_sequence)
        best = 0
        for input_seq in self.input_sequences:
            train_string = self.get_pitch_string(input_seq)
            match = SequenceMatcher(
                None, train_string, sequence_string, autojunk=False
            ).find_longest_match()
            nb_notes_common = train_string[match.a: match.a + match.size].count(" ")
            if nb_notes_common > best:
                best = nb_notes_common
        return best

    def show_conts_structure(self):
        for k in range(self.kmax):
            print(
                f"size of contexts of size {k + 1}: {len(self.prefixes_to_continuations[k])}"
            )
        # looks at the sparsity of the matrix
        order1 = self.prefixes_to_continuations[0]
        voc_size = self.voc_size()
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
        total = 0
        for k in self.viewpoints_realizations:
            total += len(self.viewpoints_realizations[k])
        print(f"average nb of vp realizations: {total / voc_size}")


if __name__ == "__main__":
    # midi_file_path = "../../data/Ravel_jeaux_deau.mid"
    # midi_file_path = "../../data/test_sequence_3notes.mid"
    # midi_file_path = "../../data/test_sequence_arpeggios.mid"
    # midi_file_path = "../../data/debussy_prelude.mid"
    # midi_file_path = "../../data/prelude_c_expressive.mid"
    # midi_file_path = "../../data/prelude_c_linear.mid"
    # midi_file_path = "../../data/partita_piano_1/pr1_1_joined.mid"
    # midi_file_path = "../../data/take6/A_quiet_place_joined.mid"
    # midi_file_path = "../../data/prelude_c_expressive.mid"
    # midi_file_path = "../../data/prelude_c.mid"
    midi_file_path = "../../data/jason examples/bach_prelude2_top.mid"
    # midi_file_path = "../../data/bach_partita_mono.midi"
    # midi_file_path = "../../data/jason examples/Sunny_Pat_Martino_solo.mid"
    # midi_file_path = "../../data/keith/train/K7_MD.mid"
    # midi_file_path = "../../../maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_03_Track03_wav--1.midi"
    t0 = time.perf_counter_ns()
    generator = Continuator2(midi_file_path, 10, transposition=False)
    # matrix = generator.get_first_order_matrix()
    # print(matrix.shape)
    # t1 = time.perf_counter_ns()
    # print(f"total time: {(t1 - t0) / 1000000}")
    # Sampling a new sequence from the  model
    generated_sequence = generator.sample_sequence(generator.get_start_vp(), length=200)
    # generated_sequence = generator.sample_zero_order(200)
    t1 = time.perf_counter_ns()
    print(f"total time: {(t1 - t0) / 1_000_000}ms")
    # print(f"generated sequence of length {len(generated_sequence)}")
    generator.save_midi(
        generated_sequence, "../../data/ctor2_output.mid", tempo=-1, sustain=False
    )
    # pmpr = generator.create_pretty_midi_pr(generated_sequence)
    # generator.plot_piano_roll(pmpr)
    # os.system("say sequence generated &")
    # print("Generated Sequence:", generated_sequence)
    # print("computing plagiarism:")
    # print(
    #     f"{generator.get_longest_subsequence_with_train(generated_sequence)} successive notes in commun with train"
    # )
    generator.show_conts_structure()
