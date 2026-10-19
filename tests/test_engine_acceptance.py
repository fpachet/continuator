"""Shared behavioral contract before archiving alternative engines."""

import pytest
from vo_regular_bp import ConstraintSet

from benchmarks.engine_audit import ENGINES, STRICT, learned_engine, phrase, scenarios, validate
from ctor.vo_regular_bp import VORegularBPContinuator
from ctor.constraints import ConstraintProblem


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("scenario", (
    "fixed", "last_note", "one_of_and_last", "prefix_last_note",
    "hidden_handoff", "fixed_end", "until_end", "prefix_until_end",
))
def test_constraint_generation_and_midi_roundtrip(kind, scenario):
    engine = learned_engine(kind)
    generate, allowed, length, handoff = scenarios(engine)[scenario]
    outputs = [validate(engine, generate(), allowed, length, handoff) for _ in range(8)]
    assert len(set(outputs)) >= 2, "branching corpus should produce varied samples"


@pytest.mark.parametrize("kind", ENGINES)
def test_impossible_known_transition_and_unreachable_end_return_none(kind):
    engine = ENGINES[kind](kmax=2)
    notes = phrase([60, 62, 64])
    engine.learn_phrase(notes, False)
    assert engine.continue_sequence(
        notes[:1], length=2, constraints={1: engine.get_viewpoint(notes[0])},
        relax_prefix_on_fail=False, relax_pos0_on_fail=False,
    ) is None
    assert engine.continue_until_end(notes[:1], min_length=1, max_length=1) is None


@pytest.mark.parametrize("kind", ENGINES)
def test_transposition_and_memory_mutation(kind):
    engine = ENGINES[kind](kmax=2)
    notes = phrase([60, 62, 64])
    engine.learn_phrase(notes, True)
    result = engine.continue_sequence([notes[0].transpose(1)], length=2, **STRICT)
    assert [n.pitch for n in engine.realize_vp_sequence(result)] == [63, 65]
    engine.clear_memory()
    assert engine.get_phrase_titles() == []
    engine.learn_phrase(phrase([70, 71, 72]), False)
    result = engine.sample_sequence(length=3, **STRICT)
    assert [n.pitch for n in engine.realize_vp_sequence(result)] == [70, 71, 72]


@pytest.mark.parametrize("stop_kind", ("symbol", "set", "predicate"))
def test_vo_regular_generalized_first_hit_stop(stop_kind):
    engine = learned_engine("vo_regular_bp")
    target = engine.get_viewpoint(phrase([67])[0])
    stop = {"symbol": target, "set": {target}, "predicate": lambda vp: isinstance(vp, tuple) and vp[0] == 67}[stop_kind]
    for _ in range(8):
        sequence = engine.continue_until(prefix=phrase([60]), stop=stop, min_length=2, max_length=32, raise_on_fail=True)
        assert 2 <= len(sequence) <= 32
        assert sequence[-1] == target and target not in sequence[:-1]
        assert engine.realize_vp_sequence(sequence)[-1].pitch == 67


def test_vo_regular_forbidden_motif_and_final_note():
    engine = learned_engine("vo_regular_bp")
    motif = tuple(engine.get_viewpoint(n) for n in phrase([60, 62, 64]))
    target = engine.get_viewpoint(phrase([60])[0])
    constraints = ConstraintSet(positional={15: {target}}, forbidden_substrings=[motif])
    outputs = []
    for _ in range(20):
        sequence = engine.sample_sequence(length=16, constraints=constraints, **STRICT)
        assert sequence[-1] == target
        assert all(tuple(sequence[i:i + 3]) != motif for i in range(len(sequence) - 2))
        outputs.append(tuple(sequence))
    assert len(set(outputs)) >= 2


@pytest.mark.xfail(strict=True, reason="VO facade forces START for memory-only sampling; classic/context-BP allow free initial symbols")
def test_vo_memory_only_can_generate_a_short_ending_suffix():
    engine = VORegularBPContinuator(kmax=2)
    engine.learn_phrase(phrase([60, 62, 64]), False)
    assert engine.sample_sequence(length=2, constraints={1: engine.get_end_vp()}) is not None


@pytest.mark.xfail(strict=True, reason="VO facade cannot emit an explicitly requested START marker, supported by classic")
def test_vo_explicit_start_marker_compatibility():
    engine = VORegularBPContinuator(kmax=2)
    engine.learn_phrase(phrase([60, 62, 64]), False)
    assert engine.sample_sequence(length=5, constraints={0: engine.get_start_vp(), 4: engine.get_end_vp()}) is not None


@pytest.mark.xfail(strict=True, reason="Final one-of containing END is incorrectly treated as an END-only constraint")
def test_vo_final_one_of_can_choose_note_instead_of_end():
    engine = VORegularBPContinuator(kmax=2)
    notes = phrase([60, 62, 64])
    engine.learn_phrase(notes, False)
    constraints = ConstraintProblem(length=3)
    constraints.at(2).one_of([engine.get_viewpoint(notes[-1]), engine.get_end_vp()])
    assert engine.sample_sequence(length=3, constraints=constraints) is not None
