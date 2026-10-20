from unittest.mock import patch

import pytest
from vo_regular_bp import ConstraintSet, LongestFeasiblePolicy, OrderStackModel

from ctor.belief_propag import NoSolutionErrorInBP
from ctor.constraints import ConstraintProblem
from ctor.sequence_model import SequenceModel


def learned():
    model = SequenceModel(kmax=3, seed=9)
    model.learn_sequences([[1, 2, 3], [1, 2, 4], [4, 2, 1]])
    return model


def test_fully_constrained_symbolic_path_with_markers():
    m = learned()
    path = [m.start_symbol, 1, 2, 3, m.end_symbol]
    assert m.sample_sequence(len(path), constraints=dict(enumerate(path)), raise_on_fail=True) == path


def test_generic_one_of_and_suffix_indexing():
    m = learned()
    problem = ConstraintProblem(length=2).at(0).one_of([3, 4])
    problem.at(1).equals(m.end_symbol)
    for _ in range(10):
        seq = m.continue_sequence([1, 2], length=2, constraints=problem, raise_on_fail=True)
        assert seq[0] in (3, 4) and seq[1] == m.end_symbol


def test_free_initial_short_suffix_and_phrase_start_are_distinct():
    m = SequenceModel(kmax=2)
    m.learn_sequence([1, 2, 3])
    assert m.sample_sequence(2, constraints={1: m.end_symbol}) == [3, m.end_symbol]
    assert m.sample_sequence(2, start_vp=m.start_symbol, constraints={1: m.end_symbol}) is None


def test_final_note_or_end_keeps_both_feasible_branches():
    m = SequenceModel(kmax=2)
    m.learn_sequence([1, 2, 3])
    c = ConstraintProblem(length=3).at(2).one_of([3, m.end_symbol])
    results = {tuple(m.sample_sequence(3, constraints=c)) for _ in range(30)}
    assert (1, 2, 3) in results and (2, 3, m.end_symbol) in results


def test_viewpoint_function_and_known_viewpoint_prefix():
    m = SequenceModel(kmax=2, viewpoint_fn=len)
    m.learn_sequence(['a', 'bb', 'ccc'])
    assert m.continue_sequence(['a'], length=2) == [2, 3]
    assert m.continue_sequence([1], length=2) == [2, 3]
    assert m.has_viewpoint(2) and not m.has_viewpoint(99)


def test_learning_always_maps_original_objects_even_when_they_equal_known_viewpoints():
    m = SequenceModel(kmax=2, viewpoint_fn=lambda value: value + 1)
    m.learn_sequence([1, 2])
    m.learn_sequence([2, 3])
    assert m.get_viewpoint(2) == 3
    assert m.order_model.counts[(3,)][4] == 1
    assert m.has_viewpoint(4)
    assert m.continue_sequence([3], length=1) == [4]


def test_tuple_symbols_and_custom_first_hit_target():
    m = SequenceModel(kmax=2)
    a, b, c = ('C', 1), ('D', 1), ('E', 2)
    m.learn_sequence([a, b, c])
    assert m.continue_until([a], stop=c, min_length=2, max_length=2) == [b, c]


def test_until_end_counts_marker_and_rejects_early_end():
    m = SequenceModel(kmax=2)
    m.learn_sequences([[1], [1, 2]])
    assert m.continue_until_end([1], min_length=2, max_length=2) == [2, m.end_symbol]
    assert m.continue_until_end([2], min_length=2, max_length=3) is None


def test_strict_constraints_even_when_legacy_relaxation_flags_are_true():
    m = SequenceModel(kmax=2)
    m.learn_sequence([1, 2, 3])
    assert m.sample_sequence(1, prefix=[1], constraints={0: 1}, relax_prefix_on_fail=True, relax_pos0_on_fail=True) is None
    with pytest.raises(NoSolutionErrorInBP):
        m.sample_sequence(1, prefix=[1], constraints={0: 1}, raise_on_fail=True)
    assert m.continue_sequence([99], length=2) is None
    assert m.sample_sequence(2, constraints={1: 99}) is None
    assert m.continue_until(stop=99) is None


@pytest.mark.parametrize('kwargs', [dict(length=-1), dict(length=2, constraints={2: 1}),
                                  dict(length=2, constraints={-1: 1}),
                                  dict(length=2, constraints=ConstraintProblem(length=3))])
def test_invalid_fixed_requests_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        learned().sample_sequence(**kwargs)


def test_empty_memory_and_zero_length():
    m = SequenceModel()
    assert m.sample_sequence(0) == []
    assert m.sample_sequence(2) is None
    assert m.continue_until_end() is None
    with pytest.raises(ValueError):
        m.continue_until_end(min_length=0)
    with pytest.raises(ValueError):
        SequenceModel(kmax=0)


def test_long_generation_without_recursion_or_early_markers():
    m = SequenceModel(kmax=2)
    m.learn_sequence([1, 2, 1, 2, 1, 2])
    seq = m.sample_sequence(1000)
    assert len(seq) == 1000 and set(seq) == {1, 2}


def test_seed_reproduces_fixed_and_until_samples():
    a, b = learned(), learned()
    for _ in range(10):
        assert a.sample_sequence(8) == b.sample_sequence(8)
        assert a.continue_until_end(min_length=2, max_length=16) == b.continue_until_end(min_length=2, max_length=16)


def test_learning_is_lazy_and_rebuilds_only_after_mutation():
    with patch.object(OrderStackModel, 'from_sequences', wraps=OrderStackModel.from_sequences) as build:
        m = learned()
        assert build.call_count == 0
        m.continue_sequence([1], length=2)
        assert build.call_count == 1
        m.continue_sequence([1], length=2)
        assert build.call_count == 1
        m.learn_sequence([1, 5, 6])
        m.continue_sequence([5], length=1)
        assert build.call_count == 2


def test_prepared_fixed_plan_reused_across_prefixes():
    m = learned()
    with patch('ctor.sequence_model.prepare_constrained_order_stack_plan', wraps=__import__('vo_regular_bp').prepare_constrained_order_stack_plan) as prepare:
        assert m.continue_sequence([1], length=1) == [2]
        assert m.continue_sequence([4], length=1) == [2]
        assert prepare.call_count == 1
        assert m.get_last_generation_diagnostics()['cache_hits'] == 1


def test_until_backend_reused_and_learning_invalidates_it():
    m = learned()
    m.continue_until_end([1], min_length=2, max_length=8)
    m.continue_until_end([1], min_length=2, max_length=8)
    before = m.get_last_generation_diagnostics()
    assert before['cache_entries'] == 1 and before['cache_hits'] == 1
    m.learn_sequence([1, 5])
    assert m.get_last_generation_diagnostics()['cache_entries'] == 0
    assert m.continue_until_end([5], min_length=1, max_length=1) == [m.end_symbol]


def test_constraint_mutation_and_policy_change_do_not_reuse_stale_plan():
    m = learned()
    c = {0: 3}
    assert m.sample_sequence(1, prefix=[2], constraints=c) == [3]
    c[0] = 4
    assert m.sample_sequence(1, prefix=[2], constraints=c) == [4]
    misses = m.cache_misses
    m.policy = LongestFeasiblePolicy()
    assert m.sample_sequence(1, prefix=[2], constraints=c) == [4]
    assert m.cache_misses == misses + 1


def test_mutating_an_old_constraint_set_cannot_corrupt_a_cached_plan():
    m = learned()
    values = {3}
    original = ConstraintSet(positional={0: values})
    assert m.sample_sequence(1, prefix=[2], constraints=original) == [3]
    values.clear()
    values.add(4)
    fresh = ConstraintSet(positional={0: {3}})
    assert m.sample_sequence(1, prefix=[2], constraints=fresh) == [3]


def test_callable_constraints_are_not_cached():
    m = learned()
    allowed = {3}
    c = ConstraintSet(positional={0: lambda s: s in allowed})
    assert m.sample_sequence(1, prefix=[2], constraints=c) == [3]
    allowed.clear()
    allowed.add(4)
    assert m.sample_sequence(1, prefix=[2], constraints=c) == [4]
    assert m.get_last_generation_diagnostics()['cache_entries'] == 0


def test_callable_stop_is_not_cached():
    m = learned()
    target = {3}
    stop = lambda s: s in target
    assert m.continue_until([2], stop=stop, min_length=1, max_length=1) == [3]
    target.clear()
    target.add(4)
    assert m.continue_until([2], stop=stop, min_length=1, max_length=1) == [4]
    assert m.get_last_generation_diagnostics()['cache_entries'] == 0


def test_cache_count_and_memory_admission_limits():
    m = SequenceModel(cache_size=1)
    m.learn_sequence([1, 2, 1, 2])
    for length in (2, 3, 4):
        m.sample_sequence(length)
        assert m.get_last_generation_diagnostics()['cache_entries'] == 1
    tiny = SequenceModel(cache_bytes=1)
    tiny.learn_sequence([1, 2, 1, 2])
    tiny.sample_sequence(4)
    assert tiny.get_last_generation_diagnostics()['cache_entries'] == 0


def test_clear_memory_discards_generation_and_caches():
    m = learned()
    m.sample_sequence(2)
    m.clear_memory()
    assert not m.last_trace
    assert m.get_last_generation_diagnostics()['cache_entries'] == 0
    assert m.sample_sequence(2) is None
    m.learn_sequence([7, 8])
    assert m.sample_sequence(2) == [7, 8]
