from pathlib import Path

from ctor.continuator import Continuator2

repo_root = Path(__file__).resolve().parents[1]
midi_file_path = repo_root / "data" / "K7_MD.mid"
generator = Continuator2(midi_file_path, 4, transposition=False)
# all_files = generator.all_midi_files_from_path("../../data/keith/train")
# generator.learn_files(all_files)
# Sampling a new sequence from the  model
# generated_sequence = generator.sample_sequence(generator.get_start_vp(), length=-1)
# print(f"generated sequence of length {len(generated_sequence)}")
# generator.save_midi(generated_sequence[1:-1], "../data/ctor2_keith_K7.mid", tempo= -1, sustain=True)

# generate the viewpoint sequence:
generated_sequence = generator.sample_sequence(length=100, constraints={99: generator.get_end_vp()})

# remove start or end viewpoint if needed
sequence_to_render = generated_sequence[0:-1]

# realize the sequence (with actual notes)
rendered_sequence = generator.realize_vp_sequence(sequence_to_render)

# save the generated sequence
generator.save_midi(rendered_sequence, repo_root / "data" / "ctor2_keith_K7.mid", tempo=-1, sustain=True)
