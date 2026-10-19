"""Run with python -m benchmarks.engine_audit --output /tmp/engine-audit.json.

This is an audit, not a musical-quality score. First-call timing includes
preparation; repeated facade calls may reuse cached inference. Sampling an
already prepared backend is also measured separately.
No timing thresholds are imposed on CI or on other machines.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import mido
import vo_regular_bp
from vo_regular_bp import ConstraintSet, prepare_constrained_order_stack

from ctor.constraints import ConstraintProblem
from ctor.vo_regular_bp import VORegularBPContinuator
from midi_stuff.mini_muse import Note


ENGINES = {
    "vo_regular_bp": VORegularBPContinuator,
}
PITCH_PHRASES = (
    (60, 62, 64, 62, 60, 65, 67, 65, 60),
    (60, 64, 67, 64, 60, 62, 65, 62, 60),
    (60, 65, 64, 62, 60, 67, 65, 64, 60),
    (60, 62, 60, 64, 60, 65, 60, 67, 60),
)
STRICT = dict(relax_prefix_on_fail=False, relax_pos0_on_fail=False, raise_on_fail=True)


def phrase(pitches):
    notes = [Note(p, 80, 1, start_time=i) for i, p in enumerate(pitches)]
    VORegularBPContinuator.set_delta_notes(notes)
    return notes


def learned_engine(kind, *, kmax=4):
    random.seed(17)
    np.random.seed(17)
    engine = ENGINES[kind](kmax=kmax)
    for pitches in PITCH_PHRASES:
        engine.learn_phrase(phrase(pitches), transposition=False)
    return engine


def scenarios(engine):
    vp = lambda p: engine.get_viewpoint(phrase([p])[0])
    end = engine.get_end_vp()
    problem = ConstraintProblem(length=16)
    problem.at(7).one_of([vp(62), vp(64)])
    problem.at(15).equals(vp(60))
    return {
        "fixed": (lambda: engine.sample_sequence(length=16, **STRICT), {}, 16, None),
        "last_note": (lambda: engine.sample_sequence(length=16, constraints={15: vp(60)}, **STRICT), {15: {vp(60)}}, 16, None),
        "one_of_and_last": (lambda: engine.sample_sequence(length=16, constraints=problem, **STRICT), problem.allowed_values_by_position, 16, None),
        "prefix_last_note": (lambda: engine.continue_sequence(phrase([60, 62]), length=16, constraints={15: vp(60)}, **STRICT), {15: {vp(60)}}, 16, vp(62)),
        "hidden_handoff": (lambda: engine.sample_sequence(start_vp=vp(62), length=16, constraints={15: vp(60)}, **STRICT), {15: {vp(60)}}, 16, vp(62)),
        "fixed_end": (lambda: engine.sample_sequence(length=17, constraints={16: end}, **STRICT), {16: {end}}, 17, None),
        "until_end": (lambda: engine.continue_until_end(min_length=4, max_length=32), {}, None, None),
        "prefix_until_end": (lambda: engine.continue_until_end(prefix=phrase([60, 62]), min_length=4, max_length=32), {}, None, vp(62)),
    }


def validate(engine, sequence, allowed, length, handoff):
    assert sequence is not None, "returned None for a feasible request"
    end, start = engine.get_end_vp(), engine.get_start_vp()
    if length is None:
        assert 4 <= len(sequence) <= 32
        assert sequence[-1] is end
    else:
        assert len(sequence) == length
    for position, values in allowed.items():
        assert sequence[position] in values, f"constraint violated at {position}"
    assert start not in sequence
    assert end not in sequence[:-1]
    if length is not None and (length - 1) not in allowed:
        assert end not in sequence
    learned = [[engine.get_viewpoint(n) for n in phrase(p)] for p in PITCH_PHRASES]
    edges = set()
    for seq in learned:
        seq = [start, *seq, end]
        edges.update(zip(seq, seq[1:]))
    path = sequence if handoff is None else [handoff, *sequence]
    assert all(pair in edges for pair in zip(path, path[1:])), "unlearned first-order transition"
    rendered = engine.realize_vp_sequence(sequence)
    expected = [vp[0] for vp in sequence if isinstance(vp, tuple)]
    assert [n.pitch for n in rendered] == expected
    assert all(np.isfinite(n.start_time) and n.duration > 0 for n in rendered)
    assert all(a.start_time <= b.start_time for a, b in zip(rendered, rendered[1:]))
    midi = engine.create_mido_sequence(rendered)
    encoded = io.BytesIO()
    midi.save(file=encoded)
    decoded = mido.MidiFile(file=io.BytesIO(encoded.getvalue()))
    assert sum(m.type == "note_on" and m.velocity > 0 for track in decoded.tracks for m in track) == len(rendered)
    assert all(m.time >= 0 for track in decoded.tracks for m in track)
    return tuple(expected)


def acceptance(samples):
    rows = []
    for kind in ENGINES:
        engine = learned_engine(kind)
        for name, (generate, allowed, length, handoff) in scenarios(engine).items():
            outputs, failures = [], []
            for _ in range(samples):
                try:
                    with redirect_stdout(io.StringIO()):
                        output = validate(engine, generate(), allowed, length, handoff)
                    outputs.append(output)
                except Exception as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")
            pairs = [(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1:] if len(a) == len(b)]
            distance = [sum(x != y for x, y in zip(a, b)) / len(a) for a, b in pairs if a]
            rows.append(dict(engine=kind, scenario=name, attempts=samples,
                             successes=len(outputs), failures=dict(Counter(failures)),
                             unique=len(set(outputs)), note_lengths=sorted(set(map(len, outputs))),
                             mean_pairwise_pitch_hamming=statistics.mean(distance) if distance else None,
                             examples=[list(s) for s in outputs[:2]]))
        print(f"acceptance {kind} complete", file=sys.stderr, flush=True)
    return rows


def compatibility_probes():
    rows = []
    for kind, cls in ENGINES.items():
        engine = cls(kmax=2)
        notes = phrase([60, 62, 64])
        engine.learn_phrase(notes, False)
        first, _middle, last = map(engine.get_viewpoint, notes)
        end, start = engine.get_end_vp(), engine.get_start_vp()
        mixed_final = ConstraintProblem(length=3)
        mixed_final.at(2).one_of([last, end])
        probes = {
            "unprompted_short_suffix": lambda: engine.sample_sequence(length=2, constraints={1: end}, **STRICT),
            "explicit_start_marker": lambda: engine.sample_sequence(length=5, constraints={0: start, 4: end}, **STRICT),
            "final_note_or_end": lambda: engine.sample_sequence(length=3, constraints=mixed_final, **STRICT),
            "unknown_prefix_default": lambda: engine.continue_sequence(phrase([99]), length=2, constraints={1: last}),
            "viewpoint_prefix": lambda: engine.continue_sequence([first], length=2, **STRICT),
            "unknown_constraint_default": lambda: engine.sample_sequence(length=2, constraints={1: (99, 1, False, False)}),
            "impossible_transition_default": lambda: engine.continue_sequence(notes[:1], length=2, constraints={1: first}, relax_prefix_on_fail=False, relax_pos0_on_fail=False),
            "unreachable_end_window": lambda: engine.continue_until_end(notes[:1], min_length=1, max_length=1),
        }
        for name, generate in probes.items():
            try:
                with redirect_stdout(io.StringIO()):
                    seq = generate()
                value = None if seq is None else ["END" if v is end else "START" if v is start else v[0] for v in seq]
                rows.append(dict(engine=kind, probe=name, result=value))
            except Exception as exc:
                rows.append(dict(engine=kind, probe=name, error=type(exc).__name__, message=str(exc)))
    return rows


def rolling_continuations(chunks=100):
    rows = []
    for kind in ENGINES:
        engine = learned_engine(kind)
        prefix = phrase([60, 62])
        outputs = []
        for _ in range(chunks):
            sequence = engine.continue_sequence(prefix, length=16, **STRICT)
            output = validate(engine, sequence, {}, 16, engine.get_viewpoint(prefix[-1]))
            outputs.append(output)
            # The shared MIDI facade contract takes played/realized Note objects.
            prefix = engine.realize_vp_sequence(sequence)[-4:]
        rows.append(dict(engine=kind, chunks=chunks, generated_notes=16 * chunks, unique_chunks=len(set(outputs))))
    return rows


def summarize_ms(values):
    first_ms = values[0] * 1000
    values = sorted(1000 * v for v in values)
    return dict(first_ms=first_ms, median_ms=statistics.median(values), p95_ms=float(np.percentile(values, 95)), max_ms=max(values), repeats=len(values))


def json_safe(value):
    """Preserve non-finite log diagnostics as strings in standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def measure(fn, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - started)
        assert result is not None
    return summarize_ms(times), result


def benchmarks(midi_path, repeats, long_until=False):
    rows = []
    for kind, cls in ENGINES.items():
        engine = cls(kmax=4)
        started = time.perf_counter()
        engine.learn_file(str(midi_path), False)
        learn_ms = 1000 * (time.perf_counter() - started)
        prefix = engine.get_phrase(0)[:4]
        target = engine.get_viewpoint(engine.get_phrase(0)[-1])
        for length in (16, 64, 256):
            timings, seq = measure(lambda: engine.continue_sequence(prefix, length=length, constraints={length - 1: target}, **STRICT), repeats)
            assert len(seq) == length and seq[-1] == target
            realization, _ = measure(lambda: engine.realize_vp_sequence(seq), repeats)
            rows.append(dict(engine=kind, scenario="bach_fixed_last_note", length=length, learn_ms=learn_ms, generation=timings, realization=realization))
        for maximum in ((32, 64, 128, 256, 500) if kind == "vo_regular_bp" and long_until else (32, 64, 128)):
            timings, seq = measure(lambda: engine.continue_until_end(prefix, min_length=4, max_length=maximum), repeats)
            assert seq[-1] is engine.get_end_vp() and 4 <= len(seq) <= maximum
            rows.append(dict(engine=kind, scenario="bach_until_end", max_length=maximum, generation=timings))
        print(f"benchmark {kind} complete", file=sys.stderr, flush=True)
    return rows


def vo_scaling(midi_path, repeats):
    reader = VORegularBPContinuator()
    material = reader.extract_notes(str(midi_path))
    rows = []
    for copies, kmax, mode in ((1, 4, "none"), (4, 4, "none"), (16, 4, "none"), (1, 1, "none"), (1, 8, "none"), (1, 4, "explicit"), (1, 4, "virtual")):
        engine = VORegularBPContinuator(kmax=kmax, transposition=mode != "none", augmentation_mode="virtual" if mode == "virtual" else "explicit")
        learning = []
        for _ in range(copies):
            started = time.perf_counter()
            engine.learn_phrase(material, transposition=mode != "none")
            learning.append(time.perf_counter() - started)
        prefix = material[:4]
        target = engine.get_viewpoint(material[-1])
        constraints = {63: target}
        cold, sequence = measure(lambda: engine.continue_sequence(prefix, length=64, constraints=constraints, **STRICT), repeats)
        realization, _ = measure(lambda: engine.realize_vp_sequence(sequence), repeats)
        prepared, backend = measure(lambda: prepare_constrained_order_stack(engine.order_model, ConstraintSet(positional={63: {target}}), length=64, prefix=tuple(engine.get_viewpoint(n) for n in prefix), policy=engine.policy), repeats)
        warm, _ = measure(lambda: backend.sample(rng=engine.rng), repeats)
        # Allocation measurement is separate so tracing does not distort timings.
        tracemalloc.start()
        engine.continue_sequence(prefix, length=64, constraints=constraints, **STRICT)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(dict(base_notes=len(material) * copies, base_phrases=copies, kmax=kmax, augmentation=mode,
                         stored_phrases=len(engine.realization_store.input_sequences),
                         learning_total_ms=1000 * sum(learning), learning_last_ms=1000 * learning[-1],
                         facade=cold, preparation=prepared, prepared_sample=warm, realization=realization,
                         python_peak_generation_mib=peak / (1024 ** 2), diagnostics=backend.diagnostics.as_dict()))
        print(f"scaling copies={copies} kmax={kmax} augmentation={mode} complete", file=sys.stderr, flush=True)
    return rows


def diverse_corpus_scaling(repeats):
    rows = []
    for count in (8, 32, 128):
        rng = random.Random(91)
        engine = VORegularBPContinuator(kmax=4)
        started = time.perf_counter()
        for _ in range(count):
            notes = phrase([rng.randrange(60, 72) for _ in range(64)])
            engine.learn_phrase(notes, False)
        learn_ms = 1000 * (time.perf_counter() - started)
        timings, seq = measure(lambda: engine.continue_sequence(notes[:4], length=64, constraints={63: engine.get_viewpoint(notes[-1])}, **STRICT), repeats)
        assert seq[-1] == engine.get_viewpoint(notes[-1])
        rows.append(dict(phrases=count, notes=64 * count, vocabulary=12, learning_total_ms=learn_ms, generation=timings, diagnostics=engine.get_last_generation_diagnostics()))
        print(f"diverse corpus phrases={count} complete", file=sys.stderr, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--midi", type=Path, default=Path("data/prelude_c.mid"))
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--long-until", action="store_true", help="Also measure VO until-end horizons 256 and 500; expensive on v0.1.1")
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    dependency_root = Path(vo_regular_bp.__file__).resolve().parent.parent
    revision = subprocess.run(["git", "-C", str(dependency_root), "rev-parse", "HEAD"], capture_output=True, text=True)
    result = dict(environment=dict(python=sys.version, platform=platform.platform(),
                                  backend_path=vo_regular_bp.__file__, backend_git_head=revision.stdout.strip() or None,
                                  midi=str(args.midi), midi_sha256=hashlib.sha256(args.midi.read_bytes()).hexdigest()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stages = {
        "acceptance": lambda: acceptance(args.samples),
        "compatibility_probes": compatibility_probes,
        "rolling_continuations": rolling_continuations,
        "benchmarks": lambda: benchmarks(args.midi, args.repeats, args.long_until),
        "vo_scaling": lambda: vo_scaling(args.midi, args.repeats),
        "diverse_corpus_scaling": lambda: diverse_corpus_scaling(args.repeats),
    }
    for name, run in stages.items():
        result[name] = run()
        args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    if any(row["failures"] for row in result["acceptance"]):
        raise SystemExit("Acceptance failures recorded; inspect output JSON")


if __name__ == "__main__":
    main()
