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


if __name__ == "__main__":
    unittest.main()
