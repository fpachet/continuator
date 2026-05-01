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

    def test_end_symbol_can_be_explicitly_constrained(self):
        model = ContextBPModel(kmax=2, seed=0)
        model.learn_sequence(["A"])

        sequence = model.sample_sequence(
            length=2,
            constraints={1: model.end_symbol},
            raise_on_fail=True,
        )

        self.assertEqual(sequence, ["A", model.end_symbol])

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
