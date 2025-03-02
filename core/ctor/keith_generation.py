import os
import pathlib

import core.ctor
from core.ctor.continuator_v3 import Continuator2

midi_file_path = "../../data/keith/train/K7_MD.mid"
generator = Continuator2(midi_file_path, 4, transposition=False)
# all_files = generator.all_midi_files_from_path("../../data/keith/train")
# generator.learn_files(all_files)
# Sampling a new sequence from the  model
generated_sequence = generator.sample_sequence(generator.get_start_vp(), length=-1)
# print(f"generated sequence of length {len(generated_sequence)}")
generator.save_midi(generated_sequence[1:-1], "../../data/keith/ctor2_keith_K7.mid", tempo= -1, sustain=True)
# print("computing plaggiarism:")
# print(
#     f"{generator.get_longest_subsequence_with_train(generated_sequence)} successive notes in commun with train"
# )
generator.show_conts_structure()
