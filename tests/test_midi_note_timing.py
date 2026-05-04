import unittest

from ctor.midi import MidiContinuatorBase, MidiRealizationStore
from midi_stuff.mini_muse import Note


class DummyMidiContinuator(MidiContinuatorBase):
    def __init__(self, store):
        self.realization_store = store


class MidiNoteTimingTest(unittest.TestCase):
    def test_overlap_viewpoints_are_derived_from_neighbor_timing(self):
        notes = [
            Note(60, 80, 2, start_time=0),
            Note(62, 80, 1, start_time=1),
            Note(64, 80, 1, start_time=3),
        ]

        MidiContinuatorBase.set_delta_notes(notes)

        self.assertEqual(
            [MidiContinuatorBase.get_viewpoint(note) for note in notes],
            [
                (60, 2, False, True),
                (62, 1, True, False),
                (64, 1, False, False),
            ],
        )

    def test_set_timing_reconstructs_interonsets_from_next_start_delta(self):
        notes = [
            Note(60, 80, 2, start_time=0),
            Note(62, 80, 1, start_time=1),
            Note(64, 80, 1, start_time=3),
        ]
        MidiContinuatorBase.set_delta_notes(notes)
        store = MidiRealizationStore(lambda note: note.pitch)
        store.learn_sequence(notes)
        continuator = DummyMidiContinuator(store)

        rendered = continuator.set_timing([(0, 0), (0, 1), (0, 2)])

        self.assertEqual([note.start_time for note in rendered], [0, 1, 3])
        self.assertEqual([note.duration for note in rendered], [2, 1, 1])

    def test_realize_vp_sequence_prefers_consecutive_realizations(self):
        store = MidiRealizationStore(lambda note: note.pitch)
        store.learn_sequence(
            [
                Note(60, 70, 1, start_time=0),
                Note(65, 70, 1, start_time=1),
                Note(62, 70, 1, start_time=2),
            ]
        )
        store.learn_sequence(
            [
                Note(60, 90, 1, start_time=0),
                Note(62, 90, 1, start_time=1),
            ]
        )
        continuator = DummyMidiContinuator(store)

        rendered = continuator.realize_vp_sequence([60, 62])

        self.assertEqual([note.velocity for note in rendered], [90, 90])

    def test_realize_vp_sequence_uses_ending_realization_when_end_marker_is_present(self):
        end_marker = object()
        store = MidiRealizationStore(lambda note: note.pitch, end_padding=end_marker)
        store.learn_sequence(
            [
                Note(62, 70, 1, start_time=0),
                Note(60, 70, 1, start_time=1),
                Note(65, 70, 1, start_time=2),
            ]
        )
        store.learn_sequence(
            [
                Note(62, 90, 1, start_time=0),
                Note(60, 90, 1, start_time=1),
            ]
        )
        continuator = DummyMidiContinuator(store)

        rendered = continuator.realize_vp_sequence([62, 60, end_marker])

        self.assertEqual([note.velocity for note in rendered], [90, 90])
        self.assertEqual(rendered[-1].next_start_delta, 0)


if __name__ == "__main__":
    unittest.main()
