import unittest
from unittest.mock import patch

from ctor.constraints import ConstraintProblem
from ctor.vo_regular_bp import VORegularBPContinuator
from midi_stuff.mini_muse import Note


def make_phrase(pitches):
    notes = [Note(pitch, 80, 1, start_time=index) for index, pitch in enumerate(pitches)]
    VORegularBPContinuator.set_delta_notes(notes)
    return notes


class VORegularBPContinuatorTest(unittest.TestCase):
    def test_generates_midi_viewpoints_with_vo_regular_bp(self):
        continuator = VORegularBPContinuator(kmax=2)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)

        sequence = continuator.continue_sequence(phrase[:2], length=1, raise_on_fail=True)

        self.assertEqual(sequence, [continuator.get_viewpoint(phrase[2])])

    def test_continuator2_defaults_to_vo_regular_bp_continuator(self):
        from ctor.continuator import Continuator2

        self.assertTrue(issubclass(Continuator2, VORegularBPContinuator))

    def test_pitch_viewpoint_constraints_work(self):
        continuator = VORegularBPContinuator(kmax=2)
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

    def test_constraint_problem_positions_are_generated_suffix_positions(self):
        continuator = VORegularBPContinuator(kmax=1)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)
        target = continuator.get_viewpoint(phrase[2])
        constraints = ConstraintProblem(length=2)
        constraints.at(1).equals(target)

        sequence = continuator.continue_sequence(
            phrase[:1],
            length=2,
            constraints=constraints,
            raise_on_fail=True,
        )

        self.assertEqual(sequence, [continuator.get_viewpoint(phrase[1]), target])

    def test_continue_until_stops_on_arbitrary_viewpoint(self):
        continuator = VORegularBPContinuator(kmax=1)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)
        target = continuator.get_viewpoint(phrase[2])

        sequence = continuator.continue_until(
            prefix=phrase[:1],
            stop=target,
            min_length=2,
            max_length=2,
            raise_on_fail=True,
        )

        self.assertEqual(sequence, [continuator.get_viewpoint(phrase[1]), target])
        self.assertEqual(len(continuator.get_last_generation_trace()), 2)
        self.assertIn("effective_order", continuator.get_last_generation_trace()[0])

    def test_until_end_uses_learned_end_symbol_and_realizes_notes(self):
        continuator = VORegularBPContinuator(kmax=2)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)

        sequence = continuator.continue_until_end(prefix=phrase[:1], min_length=3, max_length=3)
        rendered = continuator.realize_vp_sequence(sequence)

        self.assertEqual(sequence[:-1], [continuator.get_viewpoint(phrase[1]), continuator.get_viewpoint(phrase[2])])
        self.assertIs(sequence[-1], continuator.get_end_vp())
        self.assertEqual([note.pitch for note in rendered], [62, 64])

    def test_fixed_final_end_constraint_uses_first_hit_end_backend(self):
        continuator = VORegularBPContinuator(kmax=2)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)

        sequence = continuator.sample_sequence(
            prefix=None,
            start_vp=continuator.get_start_vp(),
            length=4,
            constraints={3: continuator.get_end_vp()},
            raise_on_fail=True,
        )

        self.assertEqual(
            sequence,
            [
                continuator.get_viewpoint(phrase[0]),
                continuator.get_viewpoint(phrase[1]),
                continuator.get_viewpoint(phrase[2]),
                continuator.get_end_vp(),
            ],
        )

    def test_realize_sequence_ending_at_end_symbol_uses_ending_realization(self):
        continuator = VORegularBPContinuator(kmax=2)
        phrase = [
            Note(60, 80, 1, start_time=0),
            Note(62, 80, 1, start_time=2),
            Note(60, 80, 1, start_time=3),
        ]
        continuator.set_delta_notes(phrase)
        continuator.learn_phrase(phrase, transposition=False)
        sequence = [
            continuator.get_viewpoint(phrase[1]),
            continuator.get_viewpoint(phrase[2]),
            continuator.get_end_vp(),
        ]

        with patch("random.choice", side_effect=lambda values: values[0]):
            rendered = continuator.realize_vp_sequence(sequence)

        self.assertEqual([note.pitch for note in rendered], [62, 60])
        self.assertEqual(rendered[-1].next_start_delta, 0)

    def test_explicit_transposition_materializes_transposed_realizations(self):
        continuator = VORegularBPContinuator(
            kmax=1,
            transposition_offsets=(0, 1),
        )
        phrase = make_phrase([60, 62])
        continuator.learn_phrase(phrase, transposition=True)
        prefix = [phrase[0].transpose(1)]

        sequence = continuator.continue_sequence(prefix, length=1, raise_on_fail=True)
        rendered = continuator.realize_vp_sequence(sequence)

        self.assertEqual(sequence, [(63, 1, False, False)])
        self.assertEqual([note.pitch for note in rendered], [63])

    def test_virtual_transposition_realizes_transposed_viewpoints_lazily(self):
        continuator = VORegularBPContinuator(
            kmax=1,
            transposition=True,
            augmentation_mode="virtual",
            transposition_offsets=(0, 1),
        )
        phrase = make_phrase([60, 62])
        continuator.learn_phrase(phrase, transposition=False)
        prefix = [phrase[0].transpose(1)]

        sequence = continuator.continue_sequence(prefix, length=1, raise_on_fail=True)
        rendered = continuator.realize_vp_sequence(sequence)

        self.assertEqual(sequence, [(63, 1, False, False)])
        self.assertEqual([note.pitch for note in rendered], [63])
        self.assertEqual(len(continuator.realization_store.input_sequences), 1)

    def test_virtual_transposition_can_be_enabled_from_learn_call(self):
        continuator = VORegularBPContinuator(
            kmax=1,
            augmentation_mode="virtual",
            transposition_offsets=(0, 1),
        )
        phrase = make_phrase([60, 62])
        continuator.learn_phrase(phrase, transposition=True)
        prefix = [phrase[0].transpose(1)]

        sequence = continuator.continue_sequence(prefix, length=1, raise_on_fail=True)

        self.assertEqual(sequence, [(63, 1, False, False)])
        self.assertEqual(len(continuator.realization_store.input_sequences), 1)


if __name__ == "__main__":
    unittest.main()
