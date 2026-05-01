import unittest

from ctor.constraints import ConstraintProblem
from ctor.core import ContextBPModel, NoFeasibleSequenceError


class ContextBPModelTest(unittest.TestCase):
    def test_prefix_uses_full_variable_order_context(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence([1, 2, 3])
        model.learn_sequence([9, 2, 4])

        marginals = model.symbol_marginals(length=1, prefix=[1, 2])
        sequence = model.sample_sequence(length=1, prefix=[1, 2], raise_on_fail=True)

        self.assertEqual(marginals, [{3: 1.0}])
        self.assertEqual(sequence, [3])

    def test_symbol_marginals_are_exact_for_simple_branch(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence(["A", "B"])
        model.learn_sequence(["A", "C"])

        marginals = model.symbol_marginals(length=2)

        self.assertEqual(marginals[0], {"A": 1.0})
        self.assertAlmostEqual(marginals[1]["B"], 0.5)
        self.assertAlmostEqual(marginals[1]["C"], 0.5)

    def test_constraint_problem_one_of_is_respected(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence(["A", "B"])
        model.learn_sequence(["A", "C"])
        model.learn_sequence(["A", "D"])

        constraints = ConstraintProblem(length=2)
        constraints.at(0).equals("A")
        constraints.at(1).one_of(["B", "D"])

        marginals = model.symbol_marginals(length=2, constraints=constraints)
        sequence = model.sample_sequence(length=2, constraints=constraints, raise_on_fail=True)

        self.assertEqual(marginals[0], {"A": 1.0})
        self.assertEqual(set(marginals[1]), {"B", "D"})
        self.assertNotIn("C", marginals[1])
        self.assertEqual(sequence[0], "A")
        self.assertIn(sequence[1], {"B", "D"})

    def test_constrained_inference_backs_off_to_lower_effective_order(self):
        train_seq = [1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10]
        model = ContextBPModel(kmax=10, seed=0)
        model.learn_sequence(train_seq)

        graph, _, _ = model.infer(21, constraints={9: 6, 20: model.end_symbol})
        sequence = model.sample_sequence(
            21,
            constraints={9: 6, 20: model.end_symbol},
            raise_on_fail=True,
        )

        self.assertEqual(graph.kmax, 2)
        self.assertEqual(sequence[9], 6)
        self.assertIs(sequence[-1], model.end_symbol)

    def test_end_symbol_can_be_explicitly_constrained(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence(["A"])

        sequence = model.sample_sequence(
            length=2,
            constraints={1: model.end_symbol},
            raise_on_fail=True,
        )

        self.assertEqual(sequence, ["A", model.end_symbol])

    def test_continue_until_end_respects_length_window(self):
        model = ContextBPModel(kmax=3, seed=0)
        model.learn_sequence([1, 2, 3])

        sequence = model.continue_until_end(prefix=[1], min_length=3, max_length=5)

        self.assertEqual(sequence, [2, 3, model.end_symbol])
        self.assertEqual(model.first_hit_lengths(prefix=[1], min_length=1, max_length=5), [3])

    def test_continue_until_end_does_not_choose_end_before_min_length(self):
        model = ContextBPModel(kmax=1, seed=0)
        model.learn_sequence([1])
        model.learn_sequence([1, 2])

        sequence = model.continue_until_end(prefix=[1], min_length=2, max_length=2)

        self.assertEqual(sequence, [2, model.end_symbol])

    def test_continue_until_end_returns_none_when_only_early_end_exists(self):
        model = ContextBPModel(kmax=1, seed=0)
        model.learn_sequence([1])

        sequence = model.continue_until_end(prefix=[1], min_length=2, max_length=3)

        self.assertIsNone(sequence)

    def test_continue_until_end_uses_full_prefix_context(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence([1, 2, 3])
        model.learn_sequence([9, 2, 4])

        sequence = model.continue_until_end(prefix=[1, 2], min_length=2, max_length=2)

        self.assertEqual(sequence, [3, model.end_symbol])

    def test_continue_until_end_supports_custom_target_symbol(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence(["A", "B", "C"])

        sequence = model.continue_until_end(prefix=["A"], min_length=1, max_length=2, end_symbol="C")

        self.assertEqual(sequence, ["B", "C"])

    def test_impossible_constraints_raise_when_requested(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence(["A", "B"])

        with self.assertRaises(NoFeasibleSequenceError):
            model.sample_sequence(length=1, constraints={0: "B"}, raise_on_fail=True)

    def test_constraint_problem_length_must_match_request(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence(["A"])
        constraints = ConstraintProblem(length=3)
        constraints.at(0).equals("A")

        with self.assertRaises(ValueError):
            model.symbol_marginals(length=2, constraints=constraints)


if __name__ == "__main__":
    unittest.main()
