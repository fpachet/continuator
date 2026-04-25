import unittest
from unittest.mock import patch

import numpy as np

from ctor.belief_propag import Messages, NoSolutionErrorInBP
from ctor.chain_solver import (
    NoSolutionErrorInChainSolver,
    SparseForwardBackward,
    make_unary_potentials,
)
from ctor.continuator import Continuator2
from ctor.constraints import ConstraintProblem
from ctor.variable_order_markov import Variable_order_Markov
from midi_stuff.mini_muse import Note


def bp_marginals(vo, length, constraints):
    pgm = vo.build_bp_graph(length)
    for position, viewpoint in constraints.items():
        pgm.set_value(f"x{position + 1}", vo.index_of_vp(viewpoint))
    return np.array(
        [
            Messages().marginal(pgm.variable_from_name(f"x{position + 1}"))
            for position in range(length)
        ]
    )


def forward_backward_marginals(vo, length, constraints):
    return vo.chain_marginals(length, constraints=constraints)


class SparseForwardBackwardTest(unittest.TestCase):
    def assert_marginals_match_bp(self, sequence, length, constraints):
        vo = Variable_order_Markov(sequence, None, kmax=3, seed=0)
        expected = bp_marginals(vo, length, constraints)
        actual = forward_backward_marginals(vo, length, constraints)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_matches_recursive_bp_without_constraints(self):
        self.assert_marginals_match_bp(
            sequence=[1, 2, 3, 2, 3, 4],
            length=5,
            constraints={},
        )

    def test_matches_recursive_bp_with_middle_constraint(self):
        self.assert_marginals_match_bp(
            sequence=[1, 2, 3, 2, 3, 4, 3, 4, 5],
            length=6,
            constraints={2: 3},
        )

    def test_matches_recursive_bp_with_start_and_end_padding_constraints(self):
        vo = Variable_order_Markov([1, 2, 3, 4], None, kmax=3, seed=0)
        expected = bp_marginals(vo, 6, {0: vo.start_padding, 5: vo.end_padding})
        actual = forward_backward_marginals(vo, 6, {0: vo.start_padding, 5: vo.end_padding})
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_transpose_convention_uses_previous_by_next_matrix(self):
        vo = Variable_order_Markov(["A", "B", "C"], None, kmax=2, seed=0)
        marginals = forward_backward_marginals(
            vo,
            length=5,
            constraints={0: vo.start_padding, 1: "A", 2: "B", 3: "C", 4: vo.end_padding},
        )

        expected_path = [vo.start_padding, "A", "B", "C", vo.end_padding]
        for position, viewpoint in enumerate(expected_path):
            expected = np.zeros(vo.voc_size())
            expected[vo.index_of_vp(viewpoint)] = 1.0
            np.testing.assert_allclose(marginals[position], expected, atol=1e-12, rtol=1e-12)

    def test_impossible_constraints_fail_like_recursive_bp(self):
        vo = Variable_order_Markov([1, 2, 3], None, kmax=2, seed=0)
        constraints = {0: vo.start_padding, 1: 3}

        pgm = vo.build_bp_graph(3)
        for position, viewpoint in constraints.items():
            pgm.set_value(f"x{position + 1}", vo.index_of_vp(viewpoint))
        with self.assertRaises(NoSolutionErrorInBP):
            Messages().marginal(pgm.variable_from_name("x2"))

        index_constraints = {
            position: vo.index_of_vp(viewpoint)
            for position, viewpoint in constraints.items()
        }
        unary = make_unary_potentials(
            3,
            vo.voc_size(),
            forbidden_indices={vo.index_of_vp(vo.start_padding), vo.index_of_vp(vo.end_padding)},
            constraints=index_constraints,
        )
        with self.assertRaises(NoSolutionErrorInChainSolver):
            SparseForwardBackward(vo.get_first_order_matrix()).forward_backward(unary)

    def test_iterative_solver_handles_long_chains(self):
        vo = Variable_order_Markov([1, 2, 1, 2, 1, 2], None, kmax=2, seed=0)
        marginals = vo.chain_marginals(500)
        self.assertEqual(marginals.shape, (500, vo.voc_size()))
        np.testing.assert_allclose(marginals.sum(axis=1), np.ones(500), atol=1e-12, rtol=1e-12)

    def test_constraint_problem_compiles_to_unary_masks(self):
        vo = Variable_order_Markov([1, 2, 3, 2, 3, 4], None, kmax=3, seed=0)
        problem = ConstraintProblem(length=4)
        problem.at(1).one_of([2, 3])

        unary = vo.build_unary_potentials(problem.length, constraints=problem)
        allowed = np.flatnonzero(unary[1] > 0)

        self.assertEqual(set(allowed), {vo.index_of_vp(2), vo.index_of_vp(3)})
        np.testing.assert_allclose(unary[1, allowed], np.array([0.5, 0.5]))

    def test_constraint_problem_equality_matches_legacy_bp(self):
        vo = Variable_order_Markov([1, 2, 3, 2, 3, 4], None, kmax=3, seed=0)
        problem = ConstraintProblem(length=5).at(2).equals(3)

        actual = vo.chain_marginals(problem.length, constraints=problem)
        expected = bp_marginals(vo, problem.length, problem.to_legacy_constraints())

        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_public_sampler_handles_long_chains_without_recursive_bp(self):
        vo = Variable_order_Markov([1, 2, 1, 2, 1, 2], None, kmax=2, seed=0)
        sequence = vo.sample_sequence(500)
        self.assertEqual(len(sequence), 500)

    def test_public_sampler_satisfies_constraint_problem(self):
        vo = Variable_order_Markov([1, 2, 3], None, kmax=2, seed=0)
        problem = ConstraintProblem(length=5)
        problem.at(0).equals(vo.start_padding)
        problem.at(1).one_of([1])
        problem.at(4).equals(vo.end_padding)

        sequence = vo.sample_sequence(problem.length, constraints=problem)

        self.assertIsNotNone(sequence)
        self.assertTrue(vo.sequence_satisfies_constraints(sequence, problem))
        self.assertEqual(sequence, [vo.start_padding, 1, 2, 3, vo.end_padding])

    def test_continue_sequence_indexes_constraints_over_generated_output(self):
        vo = Variable_order_Markov([1, 2, 3], None, kmax=3, seed=0)

        sequence = vo.continue_sequence(
            prefix=[1],
            length=3,
            constraints={0: 2, 2: vo.end_padding},
        )

        self.assertEqual(sequence, [2, 3, vo.end_padding])

    def test_continuator_wrapper_exposes_continue_sequence(self):
        generator = Continuator2(None, kmax=3, transposition=False)
        notes = [Note(60, 64, 1), Note(62, 64, 1), Note(64, 64, 1)]
        generator.learn_phrase(notes, transposition=False)

        sequence = generator.continue_sequence(
            prefix=[notes[0]],
            length=3,
            constraints={0: generator.get_viewpoint(notes[1]), 2: generator.get_end_vp()},
        )

        self.assertEqual(
            sequence,
            [generator.get_viewpoint(notes[1]), generator.get_viewpoint(notes[2]), generator.get_end_vp()],
        )

    def test_continue_sequence_uses_full_prefix_context_for_variable_order_choice(self):
        vo = Variable_order_Markov([1, 2, 3, 9, 2, 4], None, kmax=2, seed=0)

        with patch("ctor.variable_order_markov.random.random", return_value=0.0):
            sequence = vo.continue_sequence(prefix=[1, 2], length=1)

        self.assertEqual(sequence, [3])

    def test_reachability_to_target(self):
        vo = Variable_order_Markov([1, 2, 3], None, kmax=2, seed=0)
        solver = SparseForwardBackward(vo.get_first_order_matrix())
        reachable = solver.reachable_to_target(vo.index_of_vp(vo.end_padding), 3)

        self.assertTrue(reachable[0, vo.index_of_vp(vo.end_padding)])
        self.assertTrue(reachable[1, vo.index_of_vp(3)])
        self.assertTrue(reachable[2, vo.index_of_vp(2)])
        self.assertTrue(reachable[3, vo.index_of_vp(1)])
        self.assertFalse(reachable[1, vo.index_of_vp(1)])

    def test_first_hit_reachability_does_not_count_end_padding_loops(self):
        vo = Variable_order_Markov([1], None, kmax=1, seed=0)
        solver = SparseForwardBackward(vo.get_first_order_matrix())
        reachable = solver.first_hit_reachable_to_target(vo.index_of_vp(vo.end_padding), 3)

        self.assertTrue(reachable[1, vo.index_of_vp(1)])
        self.assertFalse(reachable[2, vo.index_of_vp(1)])
        self.assertFalse(reachable[3, vo.index_of_vp(1)])

    def test_continue_until_end_respects_length_window(self):
        vo = Variable_order_Markov([1, 2, 3], None, kmax=3, seed=0)

        sequence = vo.continue_until_end(prefix=[1], min_length=3, max_length=5)

        self.assertEqual(sequence, [2, 3, vo.end_padding])
        self.assertGreaterEqual(len(sequence), 3)
        self.assertLessEqual(len(sequence), 5)
        self.assertEqual(sequence[-1], vo.end_padding)

    def test_continue_until_end_does_not_choose_end_before_min_length(self):
        vo = Variable_order_Markov(None, None, kmax=1, seed=0)
        vo.learn_sequence([1])
        vo.learn_sequence([1, 2])

        sequence = vo.continue_until_end(prefix=[1], min_length=2, max_length=2)

        self.assertEqual(sequence, [2, vo.end_padding])

    def test_continue_until_end_returns_none_when_only_early_end_exists(self):
        vo = Variable_order_Markov([1], None, kmax=1, seed=0)

        sequence = vo.continue_until_end(prefix=[1], min_length=2, max_length=3)

        self.assertIsNone(sequence)

    def test_continue_until_end_returns_none_when_end_not_reachable_in_window(self):
        vo = Variable_order_Markov([1, 2, 3], None, kmax=3, seed=0)

        sequence = vo.continue_until_end(prefix=[1], min_length=1, max_length=2)

        self.assertIsNone(sequence)

    def test_continue_until_end_uses_full_prefix_context(self):
        vo = Variable_order_Markov(None, None, kmax=2, seed=0)
        vo.learn_sequence([1, 2, 3])
        vo.learn_sequence([9, 2, 4])

        with patch("ctor.variable_order_markov.random.random", return_value=0.0):
            sequence = vo.continue_until_end(prefix=[1, 2], min_length=2, max_length=2)

        self.assertEqual(sequence, [3, vo.end_padding])


if __name__ == "__main__":
    unittest.main()
