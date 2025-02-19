import numpy as np
import mido
import random
from collections import defaultdict
import time
class Continuator2:

    def __init__(self, midi_file, kmax=5, transposition=False):
        self.midi_file = midi_file
        self.kmax = kmax
        self.prob_to_keep_singletons = 1 / 3
        # the original sequence
        self.notes_original = self.extract_notes()
        all_notes = []
        # transpose in 12 tones
        if transposition:
            for t in range(-6, 6, 1):
                all_notes = all_notes + self.transpose_notes(self.notes_original, t)
            self.notes = all_notes
        else:
            self.notes = self.notes_original
        # the vocabulary
        self.unique_notes = list(set(self.notes))
        self.voc_size = len(self.unique_notes)
        # transform real notes to indices in the set of unique notes and vice versa
        self.note_to_idx = {note: i for i, note in enumerate(self.unique_notes)}
        self.idx_to_note = {i: note for note, i in self.note_to_idx.items()}
        # sequence of indices to unique notes
        self.seq = np.array([self.note_to_idx[note] for note in self.notes])
        self.prefixes_to_continuations = []
        self.build_vo_markov_model()

    def transpose_notes(self, notes, t):
        return [n + t for n in notes]
    def extract_notes(self):
        """Extracts the sequence of note-on events from a MIDI file."""
        mid = mido.MidiFile(self.midi_file)
        notes = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)  # Store MIDI note number
        return notes

    def build_vo_markov_model(self):
        """Builds a variable-order Markov model for max K order"""

        self.prefixes_to_continuations = np.empty(self.kmax, dtype=object)
        for k in range(self.kmax):
            prefixes_to_cont_k = {}
            for i in range(len(self.notes) - k):
                if i < k + 1:
                    continue
                current_ctx = tuple(self.notes[i-k-1:i])
                if current_ctx not in prefixes_to_cont_k:
                    prefixes_to_cont_k[current_ctx] = []
                prefixes_to_cont_k[current_ctx].append(i)
                self.prefixes_to_continuations[k] = prefixes_to_cont_k
    def get_viewpoint(self, index):
        return self.notes[index]

    def get_viewpoint_tuple(self, indices_tuple):
        vparray = [self.get_viewpoint(id) for id in indices_tuple]
        return tuple(vparray)
    def sample_sequence(self, start_note, length=50):
        """Generates a new sequence of notes from the Markov model."""
        # current_seq is a sequence of indices in the original sequence
        current_seq = [start_note]
        for _ in range(length):
            cont = self.get_continuation(current_seq)
            current_seq.append(cont)
# transforms into a sequence of notes
        return [self.notes[i] for i in current_seq]

    def get_continuation(self, current_seq):
        for k in range(self.kmax, 0, -1):
            if k > len(current_seq):
                continue
            continuations_dict = self.prefixes_to_continuations[k - 1]
            # subsequence of indices
            ctx = tuple(current_seq[-k:])
            # get the sequence of viewpoint to lookup the prefix dict
            viewpoint_ctx = self.get_viewpoint_tuple(ctx)
            if viewpoint_ctx in continuations_dict:
                all_conts = continuations_dict[viewpoint_ctx]
                # considers the number of different viewpoints, not the number of continuations
                all_cont_vp = {self.get_viewpoint(i) for i in all_conts}
                if len(all_cont_vp) == 1 and k > 0:
                    # print(f"best continuation is singleton for {k=}: {all_cont_vp}")
                    # probab to take it anyway os proportional to the number of realizations
                    print(f"best continuation is singleton for {k=}: {all_cont_vp}")
                    if random.random() >= self.prob_to_keep_singletons:
                        print(f"skipping continuation for {k=}")
                        continue
                    else:
                        print(f"not skipping continuation for {k=}")
                next_continuation = random.choice(all_conts)
                print(f"found continuation for k {k} with cont size {len(continuations_dict[viewpoint_ctx])}")
                return next_continuation
        print("no continuation found")

    def save_midi(self, sequence, output_file):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for note in sequence:
            track.append(mido.Message('note_on', note=note, velocity=64, time=200))
            track.append(mido.Message('note_off', note=note, velocity=64, time=200))
        mid.save(output_file)
        # plays the file approximatively, can be heard of Logic is open
        # with mido.open_output() as output:
        #     for note in sequence:
        #         output.send(mido.Message('note_on', note=note, velocity=64))
        #         time.sleep(0.2)
        #         output.send(mido.Message('note_off', note=note, velocity=64))

# Example usage:
midi_file_path = "../data/prelude_c.mid"
t0 = time.perf_counter_ns()
generator = Continuator2(midi_file_path, 8, transposition=True)
t1 = time.perf_counter_ns()
print(f"total time: {(t1 - t0) / 1000000}")
# Sampling a new sequence from the  model
start_note = 0  # Pick the INDEX of the first note of the original sequence
generated_sequence = generator.sample_sequence(start_note, length=100)
generator.save_midi(generated_sequence, "../data/ctor2_output.mid")
# save_midi(generated_sequence, "continuator_v2_output.mid")
print("Generated Sequence:", generated_sequence)
