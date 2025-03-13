from pathlib import Path

import numpy as np

from maxent_np.max_ent_np import MaxEnt
from maxent_np.max_entropy4 import MaxEntropyMelodyGenerator
from midi.midi_io import MidiPitchCorpus, save_midi

#
# def midi_pitch_max_ent():
#     file =
#     Kmax = 10
#     midi_io = MidiPitchCorpus(file, pitch_shifts=range(1))
#     print(midi_io.info())
#
#     me = MaxEnt(midi_io.index_seq, q=midi_io.voc_size, kmax=Kmax)
#     me.train(max_iter=10)
#     index_seq = me.sample_seq(length=760)
#     note_seq = midi_io.indices_to_notes(index_seq)
#     save_midi(note_seq, "./output.midi")
#     save_midi(index_seq, "./output-indices.midi")


class MidiMaxEnt(MaxEnt):
    def __init__(self, midi_file: Path | str, kmax: int = 10):
        self.midi_file = Path(midi_file)
        self.midi_io = MidiPitchCorpus(midi_file, pitch_shifts=range(1))
        super().__init__(self.midi_io.index_seq, q=self.midi_io.voc_size, kmax=kmax)

    def train(self, max_iter: int = 10000):
        # save_midi(
        #     self.sample_seq(length=100),
        #     "/Users/proy/Desktop/generated-max-ent-numpy-before-training.mid",
        # )
        super().train(max_iter=max_iter)

    def training_callback(self, params):
        # save_midi(
        #     self.sample_seq(length=100),
        #     f"./output-checkpoint-" f"{self.checkpoint_index}.mid",
        # )
        # print(f"Saved checkpoint #{self.checkpoint_index}")
        self.checkpoint_index += 1

    def sample_seq(self, length: int = 0, burn_in=1000):
        length = length or self.M
        index_seq = super().sample_index_seq(length=length, burn_in=burn_in)
        return self.midi_io.indices_to_notes(index_seq)

    def save_midi(self, output_file: str = "./output.midi"):
        note_seq = self.sample_seq()
        save_midi(note_seq, output_file)


if __name__ == "__main__":
    kmax = 12
    max_iter = 1000
    L = 200
    sampling_loops = 50 * L

    me = MidiMaxEnt("../data/bach_partita_mono.midi", kmax=kmax)
    # me = MidiMaxEnt("../data/test_sequence_3notes.mid", kmax=10)
    me.train(max_iter=max_iter)
    seq = me.sample_seq(length=L, burn_in=sampling_loops)
    save_midi(seq, "/Users/proy/Desktop/generated-PR.mid")
