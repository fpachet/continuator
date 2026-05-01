"""
Constrained MIDI example for Continuator2.

This example was moved out of ctor.continuator so the package module stays as a
library module. It expects data/prelude_c.mid to be available locally.
"""

import time
from pathlib import Path

from ctor.continuator import Continuator2


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    midi_file_path = repo_root / "data" / "prelude_c.mid"
    output_path = repo_root / "data" / "generated" / "constrained_prelude.mid"

    if not midi_file_path.exists():
        raise FileNotFoundError(
            f"Example input MIDI file not found: {midi_file_path}"
        )

    t0 = time.perf_counter_ns()
    generator = Continuator2(midi_file_path, 4, transposition=False)

    constraints = {
        0: generator.get_vp_for_pitch(62),
        19: generator.get_end_vp(),
    }
    generated_sequence = generator.sample_sequence(length=20, constraints=constraints)
    if generated_sequence is None:
        raise RuntimeError("No sequence could be generated for these constraints.")

    elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
    print(f"total time: {elapsed_ms}ms")

    sequence_to_render = generated_sequence[:-1]
    rendered_sequence = generator.realize_vp_sequence(sequence_to_render)
    generator.save_midi(rendered_sequence, output_path, tempo=-1, sustain=False)
    print(f"created file: {output_path}")
    generator.vom.show_conts_structure()
