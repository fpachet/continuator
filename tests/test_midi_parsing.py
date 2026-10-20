"""Regression coverage for MIDI timing and note pairing through the public facade."""

import mido
import pytest

from ctor.continuator import Continuator2
from midi_stuff.mini_muse import Note


def note_on(pitch=60, *, channel=0, time=0, velocity=80):
    return mido.Message('note_on', note=pitch, channel=channel, time=time, velocity=velocity)


def note_off(pitch=60, *, channel=0, time=0):
    return mido.Message('note_off', note=pitch, channel=channel, time=time)


def timing(notes):
    return [(n.pitch, n.velocity, n.start_time, n.duration) for n in notes]


def test_realtime_parsing_is_repeatable_and_preserves_delta_messages():
    messages = [note_on(), note_off(time=.5), note_on(62, time=.5), note_off(62, time=.5)]
    originals = [m.copy() for m in messages]
    model = Continuator2()
    expected = [(60, 80, 0, 1), (62, 80, 2, 1)]
    assert timing(model.get_phrase_from_mido(iter(messages))) == expected
    assert timing(model.get_phrase_from_mido(messages)) == expected
    assert messages == originals


def test_channels_and_repeated_pitches_are_paired_independently_in_onset_order():
    messages = [note_on(), note_on(channel=1, time=.1, velocity=90),
                note_on(time=.1, velocity=70), note_off(channel=1, time=.1),
                note_off(time=.1), note_on(time=.1, velocity=0)]
    notes = Continuator2().get_phrase_from_mido(messages)
    assert [n.velocity for n in notes] == [80, 90, 70]
    assert [n.start_time for n in notes] == pytest.approx([0, .2, .4])
    assert [n.duration for n in notes] == pytest.approx([.8, .4, .6])
    assert notes[0].overlaps_right() and notes[1].overlaps_left()


def test_duplicate_note_off_cannot_complete_the_same_note_twice():
    notes = Continuator2().get_phrase_from_mido([
        note_off(62), note_on(), note_off(time=.5), note_off(time=.5), note_on(64)])
    assert timing(notes) == [(60, 80, 0, 1)]


def test_synchronous_tracks_and_tempo_changes_keep_musical_beats(tmp_path):
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    mid.tracks.extend([
        mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=1000000),
                        mido.MetaMessage('set_tempo', tempo=250000, time=480)]),
        mido.MidiTrack([note_on(), note_off(time=960)]),
        mido.MidiTrack([note_on(64, channel=1, time=480), note_off(64, channel=1, time=480)]),
    ])
    path = tmp_path / 'parallel.mid'
    mid.save(path)
    model = Continuator2()
    notes = model.extract_notes(path)
    assert timing(notes) == [(60, 80, 0, 2), (64, 80, 1, 1)]
    assert model.tempo_msgs == [1000000, 250000]
    assert notes[0].overlaps_right() and notes[1].overlaps_left()


def test_file_parser_pairs_channels_like_realtime_parser(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    messages = [note_on(), note_on(channel=1), note_off(time=240), note_off(channel=1, time=240)]
    mid.tracks.append(mido.MidiTrack(messages))
    path = tmp_path / 'channels.mid'
    mid.save(path)
    model = Continuator2()
    assert timing(model.extract_notes(path)) == [(60, 80, 0, .5), (60, 80, 0, 1)]


def test_independent_type_two_tracks_are_rejected(tmp_path):
    mid = mido.MidiFile(type=2)
    mid.tracks.append(mido.MidiTrack([note_on(), note_off(time=480)]))
    path = tmp_path / 'independent.mid'
    mid.save(path)
    with pytest.raises(ValueError, match='independent timelines'):
        Continuator2().extract_notes(path)


@pytest.mark.parametrize('tempo', [60, 120, 180, -1])
def test_output_tempo_and_timing_roundtrip(tempo, tmp_path):
    model = Continuator2()
    model.tempo_msgs = [1000000, 500000]
    notes = [Note(60, 80, 1, 0), Note(62, 90, 1, 1)]
    mid = model.create_mido_sequence(notes, tempo=tempo)
    expected_tempo = 750000 if tempo == -1 else mido.bpm2tempo(tempo)
    assert [m.tempo for m in mid.tracks[0] if m.type == 'set_tempo'] == [expected_tempo]
    assert mid.length == pytest.approx(2 * expected_tempo / 1000000)
    path = tmp_path / 'tempo.mid'
    mid.save(path)
    assert timing(model.extract_notes(path)) == timing(notes)


def test_invalid_output_notes_and_tempo_raise():
    model = Continuator2()
    with pytest.raises(ValueError):
        model.create_mido_sequence([Note(200, 80, 1)])
    with pytest.raises(ValueError, match='tempo'):
        model.create_mido_sequence([], tempo=0)


def test_midi_tick_roundtrip_does_not_accumulate_rounding_error(tmp_path):
    model = Continuator2()
    notes = [Note(60 + i % 12, 80, 1 / 480, i / 480) for i in range(100)]
    path = tmp_path / 'ticks.mid'
    model.save_midi(notes, path)
    decoded = model.extract_notes(path)
    assert [n.start_time for n in decoded] == pytest.approx([n.start_time for n in notes])
    assert [n.duration for n in decoded] == pytest.approx([n.duration for n in notes])


def test_listener_playback_skips_metadata_and_capture_copies_messages(monkeypatch):
    from unittest.mock import Mock
    from ctor.ui.phrase_listener import MidiPhraseListener
    listener = object.__new__(MidiPhraseListener)
    listener.outport = Mock()
    listener.pending_notes = set()
    listener._stop_playing = False
    monkeypatch.setattr('ctor.ui.phrase_listener.time.sleep', lambda _: None)
    midi = Continuator2().create_mido_sequence([Note(60, 80, 1)])
    listener.play_phrase(midi)
    sent = [call.args[0] for call in listener.outport.send.call_args_list]
    assert [msg.type for msg in sent] == ['note_on', 'note_off']
    captured = note_on(time=123)
    listener.on_phrase_callback = Mock()
    listener._on_phrase_complete([(captured, .25)])
    assert captured.time == 123
    assert listener.on_phrase_callback.call_args.args[0][0].time == .25


def test_integer_file_ticks_keep_exact_one_beat_viewpoints(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    mid.tracks.append(mido.MidiTrack([
        message for _ in range(100) for message in [note_on(time=1), note_off(time=480)]
    ]))
    path = tmp_path / 'beat-grid.mid'
    mid.save(path)
    model = Continuator2()
    notes = model.extract_notes(path)
    assert all(n.duration == 1 and model.get_viewpoint(n)[1] == 1 for n in notes)
