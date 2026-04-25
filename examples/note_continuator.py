"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""
from pathlib import Path

from ctor.continuator import Continuator2

# Initialize the model
repo_root = Path(__file__).resolve().parents[1]
midi_file_path = repo_root / "data" / "prelude_c.mid"
generator = Continuator2(midi_file_path, 0, transposition=False)

# set positional constraints
constraints = {0: generator.get_vp_for_pitch(62), 99: generator.get_end_vp()}
# constraints[0] = generator.get_start_vp()

# generate the viewpoint sequence:
# generated_sequence = generator.sample_sequence(length=100, constraints=constraints)
generated_sequence = generator.sample_sequence_0(length=100)

# remove start or end viewpoint if needed
# sequence_to_render = generated_sequence
# sequence_to_render = generated_sequence[0:-1]
sequence_to_render = generated_sequence[0:-1]

# realize the sequence (with actual notes)
rendered_sequence = generator.realize_vp_sequence(sequence_to_render)

# save the generated sequence
output_path = repo_root / "data" / "prelude_0.mid"
generator.save_midi(rendered_sequence, output_path, tempo=-1)

print(f"created file: {output_path}")
