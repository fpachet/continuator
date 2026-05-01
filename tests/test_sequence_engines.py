import unittest

from ctor.engines import ClassicSequenceEngine, ContextBPSequenceEngine, make_sequence_engine


class SequenceEngineTest(unittest.TestCase):
    def test_factory_selects_engine_classes(self):
        self.assertIsInstance(make_sequence_engine("classic"), ClassicSequenceEngine)
        self.assertIsInstance(make_sequence_engine("context_bp"), ContextBPSequenceEngine)
        self.assertIsInstance(make_sequence_engine("context-bp"), ContextBPSequenceEngine)

    def test_engines_share_basic_fixed_length_api(self):
        for engine in [
            make_sequence_engine("classic", kmax=2, seed=0),
            make_sequence_engine("context_bp", kmax=2, seed=0),
        ]:
            engine.learn_sequence([1, 2, 3])

            sequence = engine.continue_sequence([1, 2], length=1, raise_on_fail=True)

            self.assertEqual(sequence, [3])

    def test_engines_share_until_end_api_shape(self):
        for engine in [
            make_sequence_engine("classic", kmax=2, seed=0),
            make_sequence_engine("context_bp", kmax=2, seed=0),
        ]:
            engine.learn_sequence([1, 2, 3])

            sequence = engine.continue_until_end(prefix=[1], min_length=3, max_length=3)

            self.assertEqual(sequence[:-1], [2, 3])
            self.assertIs(sequence[-1], engine.end_symbol)


if __name__ == "__main__":
    unittest.main()
