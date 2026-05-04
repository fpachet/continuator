import unittest

from ctor.context_bp import ContextBPContinuator, SingletonAvoidingBackoffPolicy
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

    def test_context_bp_continuator_uses_classic_singleton_avoidance(self):
        continuator = ContextBPContinuator(kmax=2)

        self.assertIsInstance(continuator.context_model.order_policy, SingletonAvoidingBackoffPolicy)

    def test_context_bp_continuator_has_no_decay_mode_control(self):
        from ctor.classic.variable_order_markov import Variable_order_Markov

        continuator = ContextBPContinuator(kmax=2)

        self.assertFalse(hasattr(continuator, "set_decay_mode"))
        self.assertFalse(hasattr(continuator, "vom"))
        self.assertNotIsInstance(continuator.realization_store, Variable_order_Markov)

    def test_context_bp_continuator_exposes_shared_phrase_memory_api(self):
        continuator = ContextBPContinuator(kmax=2)
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)

        self.assertEqual(continuator.get_phrase_titles(), ["1 phrase with 3 notes"])
        self.assertEqual([note.pitch for note in continuator.get_phrase(0)], [60, 62, 64])

    def test_context_bp_continuator_exposes_generation_trace(self):
        continuator = ContextBPContinuator(
            kmax=2,
            order_policy=SingletonAvoidingBackoffPolicy(acceptance_probability=1.0),
        )
        phrase = make_phrase([60, 62, 64])
        continuator.learn_phrase(phrase, transposition=False)

        result = continuator.continue_until_end_with_trace(prefix=phrase[:1], min_length=3, max_length=3)
        self.assertIsNotNone(result)
        sequence, trace = result
        trace_payload = continuator.get_last_generation_trace()

        self.assertEqual(sequence[-1], continuator.get_end_vp())
        self.assertEqual(len(trace), len(sequence))
        self.assertEqual(len(trace_payload), len(sequence))
        self.assertIn("policy", trace_payload[0])

    def test_context_bp_continuator_uses_free_initial_generation_without_prefix(self):
        continuator = ContextBPContinuator(kmax=1)
        phrase = make_phrase([60, 62])
        continuator.learn_phrase(phrase, transposition=False)

        sequence = continuator.sample_sequence(
            length=2,
            constraints={1: continuator.get_end_vp()},
            raise_on_fail=True,
        )
        trace = continuator.get_last_generation_trace()

        self.assertEqual(sequence, [continuator.get_viewpoint(phrase[1]), continuator.get_end_vp()])
        self.assertEqual(trace[0]["policy"], "free_initial")

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
