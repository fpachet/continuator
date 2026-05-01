import unittest

from ctor.context_bp_continuator import ContextBPContinuator
from midi_stuff.mini_muse import Note


def make_phrase(pitches):
    notes = [Note(pitch, 80, 1, start_time=index) for index, pitch in enumerate(pitches)]
    ContextBPContinuator.set_delta_notes(notes)
    return notes


class ContextBPContinuatorTest(unittest.TestCase):
    def test_generates_midi_viewpoints_with_context_bp(self):
        continuator = ContextBPContinuator(kmax=2)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)

        sequence = continuator.continue_sequence(phrase[:2], length=1, raise_on_fail=True)

        self.assertEqual(sequence, [continuator.get_viewpoint(phrase[2])])

    def test_context_bp_continuator_does_not_subclass_classic_continuator(self):
        from ctor.continuator import Continuator2

        self.assertFalse(issubclass(ContextBPContinuator, Continuator2))

    def test_until_end_uses_context_boundary_and_realizes_notes(self):
        continuator = ContextBPContinuator(kmax=3)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)

        sequence = continuator.continue_until_end(prefix=phrase[:1], min_length=3, max_length=3)
        rendered = continuator.realize_vp_sequence(sequence)

        self.assertEqual(sequence[:-1], [continuator.get_viewpoint(phrase[1]), continuator.get_viewpoint(phrase[2])])
        self.assertIs(sequence[-1], continuator.get_end_vp())
        self.assertEqual([note.pitch for note in rendered], [62, 64])

    def test_pitch_viewpoint_constraints_work_with_context_model(self):
        continuator = ContextBPContinuator(kmax=2)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)
        target = continuator.get_vp_for_pitch(64)

        sequence = continuator.continue_sequence(
            phrase[:1],
            length=2,
            constraints={1: target},
            raise_on_fail=True,
        )

        self.assertEqual(sequence[-1], target)


if __name__ == "__main__":
    unittest.main()
