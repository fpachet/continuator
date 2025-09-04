"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""
import random
import time

from muses.base.temporals import Piece

from ctor.continuator import Continuator2

from midi_stuff.mini_muse import Note

if __name__ == '__main__':
    # Initialize the model
    folder_path = "../data/bachmono"
    generator = Continuator2(None,3, transposition=False)
    t0 = time.time_ns()
    print(f"learn folder {folder_path}")
    generator.learn_folder(folder_path)
    print('learned')
    print((time.time_ns() - t0)/ 1_000_000)

    # generate the viewpoint sequence:
    generated_sequence = generator.sample_sequence(length=100)

    # remove start or end viewpoint if needed
    # sequence_to_render = generated_sequence
    # sequence_to_render = generated_sequence[0:-1]
    sequence_to_render = generated_sequence

    # realize the sequence (with actual notes)
    rendered_sequence = generator.realize_vp_sequence(sequence_to_render)

    # save the generated sequence
    generator.save_midi(rendered_sequence, "../bach_all_mono_ctor.mid", tempo=-1)
