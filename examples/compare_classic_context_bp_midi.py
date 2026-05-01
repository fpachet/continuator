"""
Compare classic and context-BP MIDI Continuator generations.

Example:

    python examples/compare_classic_context_bp_midi.py data/prelude_c.mid --samples 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ctor.classic import ClassicContinuator
from ctor.context_bp import ContextBPContinuator


def learn_material(engine, path: Path, *, transpose: bool) -> None:
    if path.is_dir():
        engine.learn_folder(str(path), transpose=transpose)
        return
    engine.learn_file(str(path), transpose)


def strip_end_symbol(engine, viewpoint_sequence: list[object]) -> list[object]:
    if viewpoint_sequence and viewpoint_sequence[-1] == engine.get_end_vp():
        return viewpoint_sequence[:-1]
    return viewpoint_sequence


def generate_viewpoints(
    engine,
    *,
    note_count: int,
    enforce_end: bool,
) -> tuple[list[object], str | None]:
    if enforce_end:
        constraints = {note_count: engine.get_end_vp()}
        viewpoint_sequence = engine.sample_sequence(
            length=note_count + 1,
            constraints=constraints,
        )
        if viewpoint_sequence is not None:
            return strip_end_symbol(engine, viewpoint_sequence), None

    viewpoint_sequence = engine.sample_sequence(length=note_count, constraints={})
    if viewpoint_sequence is None:
        return [], "no sequence"
    warning = None if not enforce_end else "relaxed end constraint"
    return strip_end_symbol(engine, viewpoint_sequence), warning


def pitch_summary(notes: Iterable[object]) -> str:
    pitches = [str(getattr(note, "pitch", "?")) for note in notes]
    return " ".join(pitches)


def print_trace(engine, *, max_steps: int = 12) -> None:
    if not hasattr(engine, "get_last_generation_trace"):
        return
    trace = engine.get_last_generation_trace()
    if not trace:
        return
    print("    trace:")
    for step in trace[:max_steps]:
        skipped = step["skipped_orders"]
        skip_label = f", skipped={skipped}" if skipped else ""
        print(
            "      "
            f"pos={step['position']} order={step['effective_order']} "
            f"symbol={step['symbol']}{skip_label}"
        )
    if len(trace) > max_steps:
        print(f"      ... {len(trace) - max_steps} more steps")


def save_if_requested(engine, notes: list[object], output_path: Path | None) -> None:
    if output_path is None or not notes:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_midi(notes, output_path, tempo=-1, sustain=False)
    print(f"    saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi_path", type=Path, help="MIDI file or folder to learn")
    parser.add_argument("--kmax", type=int, default=4, help="Maximum Markov/context order")
    parser.add_argument("--note-count", type=int, default=16, help="Generated note count")
    parser.add_argument("--samples", type=int, default=3, help="Samples per engine")
    parser.add_argument("--transpose", action="store_true", help="Learn transpositions")
    parser.add_argument("--no-end", action="store_true", help="Do not force an end marker")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional folder for generated MIDI files",
    )
    args = parser.parse_args()

    if not args.midi_path.exists():
        raise FileNotFoundError(f"MIDI path not found: {args.midi_path}")
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")

    engines = [
        ("classic", ClassicContinuator(kmax=args.kmax, transposition=args.transpose)),
        ("context_bp", ContextBPContinuator(kmax=args.kmax, transposition=args.transpose)),
    ]
    for _, engine in engines:
        learn_material(engine, args.midi_path, transpose=args.transpose)

    for engine_name, engine in engines:
        memory_size = len(getattr(engine.vom, "input_sequences", []))
        print(f"\n{engine_name} ({memory_size} learned phrases)")
        for sample_index in range(1, args.samples + 1):
            viewpoint_sequence, warning = generate_viewpoints(
                engine,
                note_count=args.note_count,
                enforce_end=not args.no_end,
            )
            if not viewpoint_sequence:
                print(f"  sample {sample_index}: no sequence")
                continue

            notes = engine.realize_vp_sequence(viewpoint_sequence)
            suffix = f" ({warning})" if warning else ""
            print(f"  sample {sample_index}:{suffix} {pitch_summary(notes)}")
            print_trace(engine)

            if args.output_dir is not None:
                output_path = args.output_dir / f"{engine_name}_{sample_index:02d}.mid"
                save_if_requested(engine, notes, output_path)


if __name__ == "__main__":
    main()
