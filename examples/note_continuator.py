"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""
from pathlib import Path

from ctor.continuator import Continuator2


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    midi_file_path = repo_root / "data" / "prelude_c.mid"
    output_path = repo_root / "data" / "generated" / "prelude_0.mid"

    if not midi_file_path.exists():
        raise FileNotFoundError(f"Example input MIDI file not found: {midi_file_path}")

    generator = Continuator2(midi_file_path, kmax=5, transposition=False)
    generated_sequence = generator.sample_sequence(length=100)
    if generated_sequence is None:
        raise RuntimeError("No sequence could be generated.")

    sequence_to_render = generated_sequence
    if sequence_to_render and sequence_to_render[-1] == generator.get_end_vp():
        sequence_to_render = sequence_to_render[:-1]

    rendered_sequence = generator.realize_vp_sequence(sequence_to_render)
    generator.save_midi(rendered_sequence, output_path, tempo=-1)

    print(f"created file: {output_path}")
