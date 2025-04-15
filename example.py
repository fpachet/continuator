from ctor.continuator import Continuator2

# Initialize the model
midi_file_path = "data/prelude_c.mid"
generator = Continuator2(midi_file_path, 4, transposition=False)

# set positional constraints
constraints = {0: generator.get_vp_for_pitch(62), 19: generator.get_end_vp()}
# constraints[0] = generator.get_start_vp()

#generate the viewpoint sequence:
generated_sequence = generator.sample_sequence(length=20, constraints=constraints)

# remove start or end viewpoint if needed
sequence_to_render = generated_sequence[0:-1]

# realize the sequence (with actual notes)
rendered_sequence = generator.realize_vp_sequence(sequence_to_render)

# save the generated sequence
generator.save_midi(rendered_sequence, "data/ctor2_output.mid", tempo=-1)