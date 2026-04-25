"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

from collections import Counter, defaultdict
from typing import Dict, Tuple, Optional

import numpy as np
import random

from ctor.belief_propag import PGM, LabeledArray, Messages, NoSolutionErrorInBP
from ctor.chain_solver import NoSolutionErrorInChainSolver, SparseForwardBackward, make_unary_potentials
from ctor.constraints import (
    ConstraintProblem,
    has_constraint_at,
    shift_constraints,
    single_value_at,
    without_constraint_at,
)


# -----------------------------------------------------------------------------
# Time-decay building blocks
# -----------------------------------------------------------------------------

class LazyExpCounter:
    """
    Per-key lazy exponential counter (exact).
    Stores (weight, last_update_step) for each key and applies decay d**Δ on read or update.

    data: key(int) -> (weight: float, last_step: int)
    """
    __slots__ = ("decay", "data", "eps")

    def __init__(self, decay: float, eps: float = 1e-12):
        self.decay = float(decay)   # in (0,1], 1.0 = no decay
        self.data: Dict[int, Tuple[float, int]] = {}
        self.eps = eps

    def update(self, key: int, step: int, incr: float = 1.0) -> None:
        w, last = self.data.get(key, (0.0, step))
        if w > 0.0 and step > last and self.decay < 1.0:
            w *= self.decay ** (step - last)
        w += incr
        self.data[key] = (w, step)

    def weights(self, now: int) -> Dict[int, float]:
        if not self.data:
            return {}
        if self.decay >= 1.0:
            # no decay, just return current weights
            return {k: w for k, (w, _) in self.data.items() if w > self.eps}
        d = self.decay
        out: Dict[int, float] = {}
        for k, (w, t) in self.data.items():
            cw = w * (d ** (now - t))
            if cw > self.eps:
                out[k] = cw
        return out


class MultiCounter:
    """
    For each (context -> next) relation, maintain:
      - full : exact counts (no decay)
      - slow : lazy exponential counter (long half-life)
      - fast : lazy exponential counter (short half-life)

    Exposes 4 "views":
      - 'full'   : full counts (no decay)
      - 'late'   : fast (recent)
      - 'middle' : slow - fast   (>=0)   -> band around intermediate ages
      - 'early'  : full - slow   (>=0)   -> emphasize older
    """
    __slots__ = ("full", "slow", "fast", "track_decays")

    def __init__(self, decay_fast: float, decay_slow: float):
        self.full: Counter[int] = Counter()
        self.track_decays = (decay_fast < 1.0) or (decay_slow < 1.0)
        self.fast = LazyExpCounter(decay_fast) if self.track_decays else None
        self.slow = LazyExpCounter(decay_slow) if self.track_decays else None

    def update(self, nxt: int, step: int, incr: float = 1.0) -> None:
        self.full[nxt] += incr
        if self.track_decays:
            self.fast.update(nxt, step, incr)
            self.slow.update(nxt, step, incr)

    def weights(self, now: int, mode: str = "full") -> Dict[int, float]:
        if (mode == "full") or (not self.track_decays):
            return {k: float(v) for k, v in self.full.items() if v > 0}

        if mode == "late":
            return self.fast.weights(now)

        if mode == "middle":
            f = self.fast.weights(now)
            s = self.slow.weights(now)
            keys = set(f) | set(s)
            out: Dict[int, float] = {}
            for k in keys:
                v = s.get(k, 0.0) - f.get(k, 0.0)
                if v > 0.0:
                    out[k] = v
            return out

        if mode == "early":
            s = self.slow.weights(now)
            keys = set(self.full) | set(s)
            out: Dict[int, float] = {}
            for k in keys:
                v = float(self.full.get(k, 0)) - s.get(k, 0.0)
                if v > 0.0:
                    out[k] = v
            return out

        raise ValueError(f"unknown mode '{mode}'")

    def nonzero_options(self, now: int, mode: str) -> int:
        return len(self.weights(now, mode))


# -----------------------------------------------------------------------------
# Sentinels
# -----------------------------------------------------------------------------

class _Start_vp:
    def __init__(self):
        pass


class _End_vp:
    def __init__(self):
        pass


# -----------------------------------------------------------------------------
# Variable-order Markov with one unified continuation dictionary + MultiCounter
# -----------------------------------------------------------------------------

class Variable_order_Markov:
    def __init__(
        self,
        sequence_of_stuff,
        vp_lambda,
        kmax: int = 5,
        # optional half-lives (in events or seconds) to enable decay views
        decay_fast_half_life: Optional[float] = None,
        decay_slow_half_life: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        # inputs
        self.viewpoint_lambda = vp_lambda
        self.start_padding = _Start_vp()
        self.end_padding = _End_vp()
        self.kmax = int(kmax)
        self.rng = random.Random(seed)

        # half-life -> decay base d = 2^(-1/H); None/<=0 => d=1 (no decay)
        def _decay_from_half_life(H: Optional[float]) -> float:
            if not H or H <= 0:
                return 1.0
            return 2.0 ** (-1.0 / float(H))

        self.decay_fast = _decay_from_half_life(decay_fast_half_life)
        # if only fast given, slow defaults ~8x slower (heuristic)
        self.decay_slow = _decay_from_half_life(
            decay_slow_half_life if decay_slow_half_life
            else (decay_fast_half_life * 8 if decay_fast_half_life else None)
        )

        # selectable "view" for reading transitions: 'full'|'late'|'middle'|'early'
        self.period_mode: str = "full"

        # global clock for lazy decay
        self.global_step: int = 0
        # if set, freeze decay at this step for reads
        self.decay_freeze_at: Optional[int] = None

        self.clear_memory()
        if sequence_of_stuff is not None:
            self.learn_sequence(sequence_of_stuff)

    # ------------------ controls for views/decay ------------------

    def set_period_mode(self, mode: str) -> None:
        assert mode in ("full", "late", "middle", "early")
        self.period_mode = mode
        self.first_order_matrix = None  # force rebuild if necessary

    def freeze_decay(self) -> None:
        self.decay_freeze_at = self.global_step
        self.first_order_matrix = None

    def unfreeze_decay(self) -> None:
        self.decay_freeze_at = None
        self.first_order_matrix = None

    # ------------------ memory / model ------------------

    def _make_counter(self) -> MultiCounter:
        return MultiCounter(self.decay_fast, self.decay_slow)

    def clear_memory(self):
        self.input_sequences = []
        self.all_unique_viewpoints = []
        self.vp2index = {}
        self.viewpoints_realizations = defaultdict(list)

        # ✅ ONE unified dictionary for all context lengths:
        # key = context tuple (length = order), value = MultiCounter
        self.ctx_to_continuations: Dict[Tuple[object, ...], MultiCounter] = defaultdict(self._make_counter)

        self.first_order_matrix = None

    def clear_first_N_phrases(self, n):
        if not self.input_sequences:
            print("nothing to remove, memory is empty")
            return
        if len(self.input_sequences) < n:
            print("nothing to remove, memory is less than " + str(n))
            return
        sequences_to_learn = self.input_sequences[n:]
        self.clear_memory()
        for seq in sequences_to_learn:
            self.learn_sequence(seq)

    def clear_last_phrase(self):
        if not self.input_sequences:
            print("nothing to remove, memory is empty")
            return
        sequences_to_learn = self.input_sequences[:-1]
        self.clear_memory()
        for seq in sequences_to_learn:
            self.learn_sequence(seq)

    def learn_sequence(self, sequence_of_stuff):
        self.first_order_matrix = None
        self.input_sequences.append(sequence_of_stuff)
        self.build_vo_markov_model(sequence_of_stuff)

    def get_input_object(self, obj_address):
        # note_address is a tuple (melody index, index in melody)
        return self.input_sequences[obj_address[0]][obj_address[1]]

    @staticmethod
    def is_starting_address(note_address):
        # 0 is the start_padding, 1 is the first actual item
        return note_address[1] == 1

    def is_ending_address(self, note_address):
        return note_address[1] == len(self.input_sequences[note_address[0]]) - 2

    def is_end_padding(self, vp):
        return vp == self.end_padding

    def voc_size(self):
        return len(self.all_unique_viewpoints)

    def random_initial_vp(self):
        mc = self.ctx_to_continuations.get((self.start_padding,), None)
        if not mc:
            raise RuntimeError("Model has no start contexts yet.")
        now = self.global_step if self.decay_freeze_at is None else self.decay_freeze_at
        w = mc.weights(now, self.period_mode) or mc.weights(now, "full")
        items, weights = zip(*w.items())
        j = self.rng.choices(items, weights=weights, k=1)[0]
        return self.all_unique_viewpoints[j]

    def random_vp_with_probs(self, probs):
        idx = np.random.choice(len(probs), p=probs)
        return self.all_unique_viewpoints[idx]

    def get_all_unique_viewpoints_except_paddings(self):
        sp, ep = self.start_padding, self.end_padding
        return [vp for vp in self.all_unique_viewpoints if vp not in (sp, ep)]

    def index_of_vp(self, vp):
        return self.vp2index[vp]

    def build_vo_markov_model(self, real_sequence):
        """Build/accumulate a VO-Markov model up to order kmax using one unified dict."""
        vp_seq = [self.start_padding] + [self.get_viewpoint(obj) for obj in real_sequence] + [self.end_padding]

        # register unique viewpoints
        for vp in vp_seq:
            if vp not in self.vp2index:
                self.vp2index[vp] = len(self.all_unique_viewpoints)
                self.all_unique_viewpoints.append(vp)

        # realizations (keep your original addressing)
        sequence_index = len(self.input_sequences) - 1
        for i, vp in enumerate(vp_seq[1:-1]):
            self.add_viewpoint_realization(i, sequence_index, vp)

        # contexts update (single pass)
        vp2idx = self.vp2index
        n = len(vp_seq)
        kmax_eff = min(self.kmax, n - 1)
        for i in range(1, n):  # i is index of "next" token
            nxt_idx = vp2idx[vp_seq[i]]
            max_k = min(kmax_eff, i)  # context length up to kmax
            for k in range(1, max_k + 1):
                ctx = tuple(vp_seq[i - k:i])
                self.ctx_to_continuations[ctx].update(nxt_idx, self.global_step)
            # advance global time once per token
            self.global_step += 1

        # ensure end->end exists (so rows aren’t empty)
        self.ctx_to_continuations[(self.end_padding,)].update(vp2idx[self.end_padding], self.global_step)
        self.global_step += 1

        self.first_order_matrix = None

    # ------------------ priors / zero-order ------------------

    def get_priors(self):
        key_counts = {key: len(continuations) for key, continuations in self.viewpoints_realizations.items()}
        total_count = sum(key_counts.values())
        priors = {key: count / total_count for key, count in key_counts.items()} if total_count > 0 else {}
        sorted_keys = self.get_all_unique_viewpoints_except_paddings()
        probability_vector = np.array([priors.get(key, 0.0) for key in sorted_keys], dtype=float)
        return probability_vector

    def sample_zero_order(self, k, constraints=None):
        priors = self.get_priors()
        return random.choices(self.get_all_unique_viewpoints_except_paddings(), weights=priors, k=k)

    # ------------------ realizations ------------------

    def add_viewpoint_realization(self, i, sequence_index, vp):
        new_address = (sequence_index, i)
        self.viewpoints_realizations[vp].append(new_address)

    # ------------------ matrices ------------------

    def get_first_order_matrix(self):
        """
        Build the order-1 transition matrix P[i, j] with the current period_mode.
        Caches only in 'full' mode (time-invariant). Decayed/band views depend on 'now'.
        """
        if self.first_order_matrix is not None and self.period_mode == "full":
            return self.first_order_matrix

        keys = self.all_unique_viewpoints
        n = len(keys)
        counts = np.zeros((n, n), dtype=float)
        now = self.global_step if self.decay_freeze_at is None else self.decay_freeze_at

        for i, vp in enumerate(keys):
            mc = self.ctx_to_continuations.get((vp,), None)
            if not mc:
                continue
            w = mc.weights(now, self.period_mode)
            if not w:
                continue
            row = counts[i]
            for j, c in w.items():
                row[j] = float(c)

        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        result = np.zeros_like(counts)
        np.divide(counts, row_sums, out=result, where=row_sums > 0)

        self.first_order_matrix = result if self.period_mode == "full" else None
        return result

    def get_first_order_matrix_no_paddings(self):
        """
        Build the order-1 transition matrix P[i, j] with the current period_mode.
        Caches only in 'full' mode (time-invariant). Decayed/band views depend on 'now'.
        """

        keys = self.get_all_unique_viewpoints_except_paddings()
        n = len(keys)
        counts = np.zeros((n, n), dtype=float)
        now = self.global_step if self.decay_freeze_at is None else self.decay_freeze_at

        for i, vp in enumerate(keys):
            mc = self.ctx_to_continuations.get((vp,), None)
            if not mc:
                continue
            w = mc.weights(now, self.period_mode)
            if not w:
                continue
            row = counts[i]
            for j, c in w.items():
                # if j > n, it means it is a start or end vp, so should skip
                if j < n:
                    row[j] = float(c)

        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        result = np.zeros_like(counts)
        np.divide(counts, row_sums, out=result, where=row_sums > 0)
        return result

    # ------------------ viewpoints & sampling ------------------

    def get_viewpoint(self, real_object):
        if self.viewpoint_lambda is None:
            return real_object
        return self.viewpoint_lambda(real_object)

    def get_realizations_for_vp(self, vp):
        return self.viewpoints_realizations[vp]

    def sample_sequence(
            self,
            length: int,
            prefix=None,
            constraints: Optional[Dict[int, object]] = None,
            *,
            relax_prefix_on_fail: bool = True,
            relax_pos0_on_fail: bool = True,
            raise_on_fail: bool = False,
    ):
        """
        Sample a viewpoint sequence of given `length`.

        Parameters
        ----------
        prefix : sequence of notes/viewpoints forming the *last played* phrase.
                 Used as conditioning context only; it is not included in the return value.
        constraints : dict[int -> viewpoint]
            Hard constraints at generated positions (0-based). Values are viewpoint objects
            (NOT indices).

        Relaxation logic (if NoSolutionError):
          1) Try with all constraints + full prefix context.
          2) If fail and relax_prefix_on_fail: retry with prefix context removed.
          3) If still fail and relax_pos0_on_fail and 0 in constraints: drop the hard constraint at pos 0, keep bias off.
        """

        constraints = constraints or {}

        if prefix is not None:
            return self.continue_sequence(
                prefix,
                length,
                constraints=constraints,
                relax_prefix_on_fail=relax_prefix_on_fail,
                relax_pos0_on_fail=relax_pos0_on_fail,
                raise_on_fail=raise_on_fail,
            )

        try:
            return self.sample_vp_sequence_with_chain_solver(length, None, constraints=constraints)
        except NoSolutionErrorInChainSolver as e:
            if raise_on_fail:
                raise NoSolutionErrorInBP("No solution for constraints.") from e
            return None

    def continue_sequence(
            self,
            prefix,
            length: int,
            constraints: Optional[Dict[int, object]] = None,
            *,
            relax_prefix_on_fail: bool = True,
            relax_pos0_on_fail: bool = True,
            raise_on_fail: bool = False,
    ):
        """
        Generate a continuation after `prefix`.

        The returned sequence has exactly `length` generated viewpoints. The
        prefix is conditioning context only and is not included in the return
        value. Constraints are indexed over the returned continuation.
        """
        constraints = constraints or {}
        prefix_vps = [self.get_viewpoint(item) for item in prefix]
        if not prefix_vps:
            return self.sample_sequence(
                length,
                constraints=constraints,
                relax_prefix_on_fail=relax_prefix_on_fail,
                relax_pos0_on_fail=relax_pos0_on_fail,
                raise_on_fail=raise_on_fail,
            )

        def _sample(graph_length, first_vp, active_constraints, context_prefix=None):
            try:
                return self.sample_vp_sequence_with_chain_solver(
                    graph_length,
                    first_vp,
                    constraints=active_constraints,
                    context_prefix=context_prefix,
                )
            except NoSolutionErrorInChainSolver as e:
                return None

        # Attempt 1: all constraints + prefix context. The chain contains an
        # extra fixed first variable, so continuation constraints shift by one.
        translated_constraints = shift_constraints(constraints, 1)
        start_vp = prefix_vps[-1]
        last_error = None
        try:
            seq = _sample(length + 1, start_vp, translated_constraints, prefix_vps)
            if seq is not None:
                return seq[1:]
            # returns the sequence except the prefix
        except NoSolutionErrorInChainSolver as e:
            last_error = e

        print('give up prefix constraint (continuation)')
        # Attempt 2: relax the prefix bias only
        if relax_prefix_on_fail:
            try:
                translated_constraints = shift_constraints(constraints, 1)
                constrained_start = single_value_at(constraints, 1)
                start_vp = constrained_start if constrained_start is not None else self.start_padding
                seq = _sample(length + 1, start_vp, translated_constraints)
                if seq is not None:
                    return seq[1:] # returns the sequence except the startvp
            except NoSolutionErrorInChainSolver as e:
                last_error = e

        print('give up start sequence constraint')
        # Attempt 3: also drop the hard constraint at position 0 (if any)
        if relax_pos0_on_fail and has_constraint_at(constraints, 0):
            try:
                loosened = without_constraint_at(constraints, 0)
                seq = _sample(length, None, loosened)
                if seq is not None:
                    return seq
            except NoSolutionErrorInChainSolver as e:
                last_error = e

        if raise_on_fail:
            raise NoSolutionErrorInBP("No solution after relaxing prefix bias and pos0 constraint.") from last_error
        return None

    def continue_until_end(
            self,
            prefix=None,
            *,
            min_length: int = 1,
            max_length: int = 64,
            end_vp=None,
    ):
        """
        Generate a variable-length sequence that reaches `end_vp`.

        The prefix is conditioning context only and is not included in the
        returned sequence. The returned sequence includes `end_vp`.
        """
        if min_length < 1:
            raise ValueError("min_length must be at least 1")
        if max_length < min_length:
            raise ValueError("max_length must be greater than or equal to min_length")

        end_vp = self.end_padding if end_vp is None else end_vp
        end_index = self.index_of_vp(end_vp)
        first_order_matrix = self.get_first_order_matrix()
        solver = SparseForwardBackward(first_order_matrix)
        reachable = solver.first_hit_reachable_to_target(end_index, max_length)

        if prefix:
            context = [self.get_viewpoint(item) for item in prefix]
            current_vp = context[-1]
        else:
            current_vp = self.random_initial_vp()
            context = [current_vp]

        current_index = self.index_of_vp(current_vp)
        if not solver.can_reach_between(reachable, current_index, min_length, max_length):
            return None

        generated = []
        for step in range(max_length):
            remaining_after_next = max_length - (step + 1)
            min_after_next = max(0, min_length - (step + 1))
            current_index = self.index_of_vp(context[-1])
            row = first_order_matrix[current_index].copy()

            for next_index in range(len(row)):
                if row[next_index] <= 0:
                    continue
                if next_index == end_index and step + 1 < min_length:
                    row[next_index] = 0.0
                    continue
                if not solver.can_reach_between(reachable, next_index, min_after_next, remaining_after_next):
                    row[next_index] = 0.0

            cont = self.get_continuation_with_bp(context, row)
            if cont is None:
                total = row.sum()
                if total <= 0:
                    return None
                cont = self.random_vp_with_probs(row / total)

            generated.append(cont)
            context.append(cont)

            if cont == end_vp and len(generated) >= min_length:
                return generated

        return None

    # length of bp graph is length + 2: plus the start (possibly the end of an existing sequence) and plus the end viewpoint
    def build_bp_graph(self, length):
        string = ""
        for i in range(length):
            string = string + "p(x" + str(i + 1) + ")"
        for i in range(2, length + 1):
            string = string + "p(x" + str(i) + "|x" + str(i - 1) + ")"
        pgm = PGM.from_string(string)
        mat = LabeledArray(np.array(self.get_first_order_matrix()).transpose(), ["x2", "x1"], )
        m = self.voc_size()
        data_dict = {}
        for i in range(length):
            variable_dist = np.random.uniform(1 / m, 1 / m, m)
            # should avoid start and end values
            variable_dist[self.index_of_vp(self.start_padding)] = 0
            variable_dist[self.index_of_vp(self.end_padding)] = 0
            variable_dist /= variable_dist.sum()
            data_dict["p(x" + str(i + 1) + ")"] = LabeledArray(np.array(variable_dist), ["x" + str(i + 1)])
            data_dict["p(x" + str(i + 2) + "|x" + str(i + 1) + ")"] = LabeledArray(
                mat.array, ["x" + str(i + 2), "x" + str(i + 1)]
            )
        pgm.set_data(data_dict)
        return pgm

    def build_unary_potentials(self, length, constraints=None):
        constraints = constraints or {}
        allowed_indices_by_position = {}
        if isinstance(constraints, ConstraintProblem):
            index_constraints = {}
            allowed_indices_by_position = constraints.to_allowed_indices(self.vp2index)
        else:
            index_constraints = {
                position: self.index_of_vp(viewpoint)
                for position, viewpoint in constraints.items()
            }
        forbidden = {
            self.index_of_vp(self.start_padding),
            self.index_of_vp(self.end_padding),
        }
        return make_unary_potentials(
            length,
            self.voc_size(),
            forbidden_indices=forbidden,
            allowed_indices_by_position=allowed_indices_by_position,
            constraints=index_constraints,
        )

    def chain_marginals(self, length, constraints=None):
        unary_potentials = self.build_unary_potentials(length, constraints=constraints)
        solver = SparseForwardBackward(self.get_first_order_matrix())
        return solver.forward_backward(unary_potentials).marginals

    def sample_vp_sequence_with_bp(self, length, first_vp, pgm):
        if length < 0:
            print(f"impossible to sample a sequence of length {length}")
            return None

        if first_vp is not None:
            current_seq = [first_vp]
        else:
            try:
                marginal_1 = Messages().marginal(pgm.variable_from_name('x1'))
                vp = self.random_vp_with_probs(marginal_1)
                current_seq = [vp]
            except NoSolutionErrorInBP:
                return None
        try:
            pgm.set_value('x1', self.index_of_vp(current_seq[0]))
        except NoSolutionErrorInBP:
            return None
        # generate the rest of the sequence
        first_order_matrix = self.get_first_order_matrix()
        for i in range(length - 1):
            pgm_variable = pgm.variable_from_name('x' + str(i + 2))
            try:
                marginal_i = Messages().marginal(pgm_variable)
            except NoSolutionErrorInBP:
                return None
            markov_proba = first_order_matrix[self.index_of_vp(current_seq[-1])]
            product_proba = marginal_i * markov_proba
            cont = self.get_continuation_with_bp(current_seq, product_proba)
            if cont is None:
                cont = self.random_initial_vp()
            current_seq.append(cont)
            pgm.set_value('x' + str(i + 2), self.index_of_vp(cont))
        return current_seq

    def sample_vp_sequence_with_chain_solver(self, length, first_vp=None, constraints=None, context_prefix=None):
        if length < 0:
            print(f"impossible to sample a sequence of length {length}")
            return None
        if length == 0:
            return []

        unary_potentials = self.build_unary_potentials(length, constraints=constraints)
        if first_vp is not None:
            unary_potentials[0] = 0.0
            unary_potentials[0, self.index_of_vp(first_vp)] = 1.0

        first_order_matrix = self.get_first_order_matrix()
        solver = SparseForwardBackward(first_order_matrix)
        inference = solver.forward_backward(unary_potentials)

        if first_vp is not None:
            current_seq = [first_vp]
        else:
            current_seq = [self.random_vp_with_probs(inference.marginals[0])]

        sampling_context = list(context_prefix or current_seq)
        if not sampling_context:
            sampling_context = list(current_seq)
        if sampling_context[-1] != current_seq[-1]:
            sampling_context.append(current_seq[-1])

        for i in range(length - 1):
            next_position = i + 1
            markov_proba = first_order_matrix[self.index_of_vp(current_seq[-1])]
            suffix_proba = unary_potentials[next_position] * inference.backward[next_position]
            feasible_proba = markov_proba * suffix_proba
            cont = self.get_continuation_with_bp(sampling_context, feasible_proba)
            if cont is None:
                total = feasible_proba.sum()
                if total <= 0:
                    return None
                cont = self.random_vp_with_probs(feasible_proba / total)
            current_seq.append(cont)
            sampling_context.append(cont)

        if not self.sequence_satisfies_constraints(current_seq, constraints):
            return None
        return current_seq

    @staticmethod
    def sequence_satisfies_constraints(sequence, constraints=None):
        if constraints is None:
            return True
        if isinstance(constraints, ConstraintProblem):
            for position, allowed_values in constraints.allowed_values_by_position.items():
                if position < 0 or position >= len(sequence):
                    return False
                if sequence[position] not in allowed_values:
                    return False
            return True
        for position, value in constraints.items():
            if position < 0 or position >= len(sequence):
                return False
            if sequence[position] != value:
                return False
        return True

    def sample_vp_sequence(self, first_vp, length, last_vp):
        current_seq = [first_vp]
        if length >= 0:
            for _ in range(length):
                cont = self.get_continuation(current_seq)
                if cont is None:
                    cont = self.random_initial_vp()
                current_seq.append(cont)
            return current_seq
        while True:
            cont = self.get_continuation(current_seq)
            if cont is None:
                cont = self.random_initial_vp()
            if cont == last_vp:
                if cont != self.end_padding:
                    current_seq.append(cont)
                return current_seq
            current_seq.append(cont)

    def get_continuation(self, current_seq):
        vp_to_skip = None
        now = self.global_step if self.decay_freeze_at is None else self.decay_freeze_at
        for k in range(self.kmax, 0, -1):
            if k > len(current_seq):
                continue
            ctx = tuple(current_seq[-k:])
            mc = self.ctx_to_continuations.get(ctx)
            if not mc:
                continue
            w = mc.weights(now, self.period_mode)
            if not w:
                continue
            # keep your singleton-skip heuristic
            if len(w) == 1 and k > 1:
                if random.random() > (1 / (k + 1)):
                    vp_to_skip = next(iter(w))
                    continue
                else:
                    vp_to_skip = None
            if vp_to_skip is not None and k > 1 and vp_to_skip in w:
                w = {kk: vv for kk, vv in w.items() if kk != vp_to_skip}
                if not w:
                    continue
            items, weights = zip(*w.items())
            j = self.rng.choices(items, weights=weights, k=1)[0]
            return self.all_unique_viewpoints[j]
        print("no continuation found")
        return None

    def get_continuation_with_bp(self, current_seq, probs):
        vp_to_skip = None
        now = self.global_step if self.decay_freeze_at is None else self.decay_freeze_at
        for k in range(self.kmax, 0, -1):
            if k > len(current_seq):
                continue
            ctx = tuple(current_seq[-k:])
            mc = self.ctx_to_continuations.get(ctx)
            if not mc:
                continue
            w = mc.weights(now, self.period_mode)
            if not w:
                continue
            # mask with BP posterior > 0
            mask = [p > 0 for p in probs]
            w = {j: v for j, v in w.items() if mask[j]}
            if not w:
                continue
            if len(w) == 1 and k > 1:
                if random.random() > (1 / (k + 1)):
                    vp_to_skip = next(iter(w))
                    continue
                else:
                    vp_to_skip = None
            if vp_to_skip is not None and k > 1 and vp_to_skip in w:
                del w[vp_to_skip]
                if not w:
                    continue
            items, weights = zip(*w.items())
            j = self.rng.choices(items, weights=weights, k=1)[0]
            return self.all_unique_viewpoints[j]
        print("no continuation found")
        return None

    def show_conts_structure(self):
        # counts per context length
        by_len = defaultdict(int)
        for ctx in self.ctx_to_continuations.keys():
            by_len[len(ctx)] += 1
        for k in range(1, self.kmax + 1):
            print(f"size of contexts of size {k}: {by_len.get(k, 0)}")

        # sparsity of order-1
        voc_size = self.voc_size()
        now = self.global_step if self.decay_freeze_at is None else self.decay_freeze_at
        min_size = voc_size
        max_size = 0
        for vp in self.all_unique_viewpoints:
            mc = self.ctx_to_continuations.get((vp,), None)
            csz = mc.nonzero_options(now, self.period_mode) if mc else 0
            min_size = min(min_size, csz)
            max_size = max(max_size, csz)
        print(f"voc size: {voc_size}")
        print(f"min order 1 size: {min_size}, max: {max_size}")
        total = 0
        for k in self.viewpoints_realizations:
            total += len(self.viewpoints_realizations[k])
        print(f"average nb of vp realizations: {total / voc_size if voc_size else 0.0}")

if __name__ == '__main__':
    # computes chord sequences of length 8 starting and ending with, say, C and with a F#7 in the middle
    with open('../data/chord_sequences.txt', 'r') as file:
        seqs = file.readlines()[:200]
    seqs = [seq.split(';')[1:-1] for seq in seqs]
    seqs = [[chord.strip() for chord in seq] for seq in seqs]
    vo = Variable_order_Markov(None, None, kmax=3)
    for seq in seqs:
        vo.learn_sequence(seq)

    length = 8
    for i in range(20):
        seq = vo.sample_sequence(length, constraints={0: vo.get_viewpoint('C'), int(length/2): vo.get_viewpoint('F#7'), length - 1: vo.get_viewpoint('C')})
        result = ' '.join(seq)
        print(result)
