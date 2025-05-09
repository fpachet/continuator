"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

from ctor.continuator import Continuator2

# Initialize the model
midi_file_path = "../data/prelude_c.mid"
generator = Continuator2(midi_file_path, 4, transposition=True)

# set positional constraints
constraints = {0: generator.get_vp_for_pitch(62), 19: generator.get_end_vp()}
# constraints[0] = generator.get_start_vp()

# generate the viewpoint sequence:
generated_sequence = generator.sample_sequence(length=20, constraints=constraints)

# remove start or end viewpoint if needed
sequence_to_render = generated_sequence[0:-1]

# realize the sequence (with actual notes)
rendered_sequence = generator.realize_vp_sequence(sequence_to_render)

# save the generated sequence
generator.save_midi(rendered_sequence, "../data/output.mid", tempo=-1)

print("created file: ../data/output.mid")