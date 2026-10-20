"""Memory semantics and optional contract test against the sibling frontend."""

from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import mido
import pytest

from benchmarks.engine_audit import phrase
from ctor.continuator import Continuator2
from ctor.constraints import ConstraintProblem
from ctor.vo_regular_bp import VORegularBPContinuator


def test_exact_duplicate_phrases_keep_learning_weight_but_share_realization_domains():
    m = Continuator2()
    notes = phrase([60, 62, 64])
    for _ in range(16):
        m.learn_phrase(notes, False)
    viewpoints = [m.get_viewpoint(n) for n in notes]
    assert len(m.get_phrase_titles()) == 16
    assert m.order_model.counts[(viewpoints[0],)][viewpoints[1]] == 16
    assert all(len(domain) == 1 for domain in m._realization_domains(viewpoints))
    assert [n.pitch for n in m.realize_vp_sequence(viewpoints)] == [60, 62, 64]
    m.clear_first_n_phrases(1)
    assert len(m.get_phrase_titles()) == 15
    assert all(len(domain) == 1 for domain in m._realization_domains(viewpoints))


def test_different_velocities_are_not_deduplicated():
    m = Continuator2()
    a, b = phrase([60, 62]), phrase([60, 62])
    b[1].velocity = 99
    m.learn_phrase(a, False)
    m.learn_phrase(b, False)
    assert len(m._realization_domains([m.get_viewpoint(a[-1])])[0]) == 2


def test_midi_mutations_clear_model_cache_and_generation_trace():
    m = Continuator2()
    m.learn_phrase(phrase([60, 62, 64]), False)
    m.sample_sequence(length=3)
    assert m.get_last_generation_trace()
    m.clear_last_phrase()
    assert m.get_last_generation_trace() == []
    assert m.sample_sequence(length=3) is None
    m.learn_phrase(phrase([70, 71, 72]), False)
    assert [n.pitch for n in m.realize_vp_sequence(m.sample_sequence(length=3))] == [70, 71, 72]


def test_forget_keeps_only_newest_phrases_and_rebuilds_once():
    m = Continuator2()
    m.set_forget(True)
    m.set_keep_last(2)
    with patch.object(m.sequence_model, '_model_factory', wraps=m.sequence_model._model_factory) as build:
        for pitches in ([60, 62], [70, 72], [80, 82]):
            m.learn_phrase(phrase(pitches), False)
        assert build.call_count == 0
        assert len(m.get_phrase_titles()) == 2
        seq = m.sample_sequence(length=2)
        assert {v[0] for v in seq} <= {70, 72, 80, 82}
        assert build.call_count == 1


@pytest.mark.parametrize('augmentation_mode', ['explicit', 'virtual'])
def test_augmented_free_initial_generation_and_cache_invalidation(augmentation_mode):
    m = VORegularBPContinuator(kmax=2, transposition=True, augmentation_mode=augmentation_mode,
                              transposition_offsets=(0, 1))
    m.learn_phrase(phrase([60, 62, 64]), True)
    for _ in range(8):
        seq = m.sample_sequence(length=2, constraints={1: m.get_end_vp()}, raise_on_fail=True)
        assert m.realize_vp_sequence(seq)[0].pitch in (64, 65)
    assert m.get_last_generation_diagnostics()['cache_entries'] == 1
    m.set_transpose(False)
    assert m.get_last_generation_diagnostics()['cache_entries'] == 0


def test_declared_constraint_length_is_validated_by_midi_facade():
    m = Continuator2()
    with pytest.raises(ValueError):
        m.sample_sequence(length=2, constraints=ConstraintProblem(length=3))


def test_file_roundtrip_from_public_facade(tmp_path):
    m = Continuator2()
    m.learn_phrase(phrase([60, 62, 64]), False)
    seq = m.sample_sequence(length=3, start_vp=m.get_start_vp(), raise_on_fail=True)
    path = tmp_path / 'continuation.mid'
    m.save_midi(m.realize_vp_sequence(seq), path)
    decoded = mido.MidiFile(path)
    assert [msg.note for track in decoded.tracks for msg in track if msg.type == 'note_on'] == [60, 62, 64]


def test_sibling_frontend_single_engine_contract():
    backend = Path(__file__).resolve().parents[2] / 'continuator_front' / 'backend'
    if not (backend / 'app' / 'continuator_adapter.py').exists():
        pytest.skip('Optional sibling frontend checkout is not present')
    pytest.importorskip('pydantic')
    program = '''
import sys
sys.path.insert(0, sys.argv[1])
from app.continuator_adapter import ContinuatorSessionEngine
from app.schemas import MidiEvent
from ctor.vo_regular_bp import VORegularBPContinuator
events = []
for pitch in [60, 62, 64]:
    events.extend([MidiEvent(type='note_on', note=pitch, velocity=80, delta_seconds=0),
                   MidiEvent(type='note_off', note=pitch, velocity=0, delta_seconds=0.5)])
session = ContinuatorSessionEngine()
assert isinstance(session._continuator, VORegularBPContinuator)
session.learn_phrase_events(events)
payload, status, trace, message = session.generate_phrase(note_count=3)
assert payload.note_count == 3 and status.end.applied and trace
# The first returned note is after the played C; the hidden handoff is excluded.
_, continuation, status, trace, message = session.continue_phrase(
    events[:2], learn_input=False, continuation_note_count=2)
assert [note.pitch for note in continuation.notes] == [62, 64]
assert status.start.applied and status.end.applied and trace
assert '<svg' in session.render_graph_svg()
# Exercise the same repeated API/handoff pattern used by frontend streaming.
from benchmarks.engine_audit import PITCH_PHRASES
session = ContinuatorSessionEngine()
for pitches in PITCH_PHRASES:
    material = []
    for pitch in pitches:
        material.extend([MidiEvent(type='note_on', note=pitch, velocity=80, delta_seconds=0),
                         MidiEvent(type='note_off', note=pitch, velocity=0, delta_seconds=0.5)])
    session.learn_phrase_events(material)
payload, _, _, _ = session.generate_phrase(note_count=16, enforce_end_constraint=False)
for _ in range(20):
    _, payload, status, trace, message = session.continue_phrase(
        payload.events, learn_input=False, continuation_note_count=16,
        enforce_end_constraint=False, handoff_viewpoint=payload.handoff_viewpoint)
    assert payload.note_count == 16 and status.start.applied and trace
print('Frontend single engine, handoff, END and graph passed')
'''
    result = subprocess.run([sys.executable, '-c', program, str(backend)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
