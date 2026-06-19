import inspect
import unittest


class PublicApiTest(unittest.TestCase):
    def test_continuator2_import_and_generation_signatures(self):
        from ctor.classic import ClassicContinuator as PackageClassicContinuator
        from ctor.continuator import ClassicContinuator, Continuator2
        from ctor.midi import MidiContinuatorBase
        from ctor.vo_regular_bp import VORegularBPContinuator

        self.assertIs(ClassicContinuator, PackageClassicContinuator)
        self.assertTrue(issubclass(ClassicContinuator, MidiContinuatorBase))
        self.assertTrue(issubclass(Continuator2, VORegularBPContinuator))
        self.assertFalse(issubclass(Continuator2, PackageClassicContinuator))

        init_sig = inspect.signature(Continuator2)
        self.assertIn("midi_file", init_sig.parameters)
        self.assertIn("kmax", init_sig.parameters)
        self.assertIn("transposition", init_sig.parameters)

        sample_sig = inspect.signature(Continuator2.sample_sequence)
        for name in [
            "prefix",
            "length",
            "constraints",
            "start_vp",
            "relax_prefix_on_fail",
            "relax_pos0_on_fail",
            "raise_on_fail",
        ]:
            self.assertIn(name, sample_sig.parameters)

        continue_sig = inspect.signature(Continuator2.continue_sequence)
        for name in [
            "prefix",
            "length",
            "constraints",
            "relax_prefix_on_fail",
            "relax_pos0_on_fail",
            "raise_on_fail",
        ]:
            self.assertIn(name, continue_sig.parameters)

        until_end_sig = inspect.signature(Continuator2.continue_until_end)
        for name in ["prefix", "min_length", "max_length", "end_vp"]:
            self.assertIn(name, until_end_sig.parameters)

        decay_sig = inspect.signature(Continuator2.set_decay_mode)
        self.assertIn("choice", decay_sig.parameters)

    def test_core_public_imports(self):
        from ctor.belief_propag import NoSolutionErrorInBP
        from ctor.chain_solver import SparseForwardBackward
        from ctor.classic import Variable_order_Markov as PackageVariableOrderMarkov
        from ctor.continuator import ClassicContinuator, Continuator2
        from ctor.constraints import ConstraintProblem
        from ctor.context_bp_continuator import ContextBPContinuator
        from ctor.vo_regular_bp_continuator import VORegularBPContinuator
        from ctor.context_bp import (
            ContextBPModel,
            LongestFeasiblePolicy,
            PolicyDecision,
            SingletonAvoidingBackoffPolicy,
        )
        from ctor.engines import make_sequence_engine
        from ctor.midi import MidiContinuatorBase
        from ctor.variable_order_markov import Variable_order_Markov

        self.assertTrue(issubclass(NoSolutionErrorInBP, Exception))
        self.assertIsNotNone(SparseForwardBackward)
        self.assertIsNotNone(ClassicContinuator)
        self.assertTrue(issubclass(Continuator2, VORegularBPContinuator))
        self.assertIsNotNone(ConstraintProblem)
        self.assertIsNotNone(ContextBPContinuator)
        self.assertIsNotNone(VORegularBPContinuator)
        self.assertIsNotNone(make_sequence_engine)
        self.assertIsNotNone(MidiContinuatorBase)
        self.assertIsNotNone(Variable_order_Markov)
        self.assertIs(Variable_order_Markov, PackageVariableOrderMarkov)
        self.assertIsNotNone(ContextBPModel)
        self.assertIsNotNone(LongestFeasiblePolicy)
        self.assertIsNotNone(PolicyDecision)
        self.assertIsNotNone(SingletonAvoidingBackoffPolicy)

    def test_ui_package_import_does_not_eagerly_import_gradio_app(self):
        import ctor.ui

        self.assertEqual(ctor.ui.__all__, ["Continuator_gradio"])


if __name__ == "__main__":
    unittest.main()
