"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

from collections import Counter, defaultdict
from typing import Dict, Tuple, Optional

import numpy as np
import random
from difflib import SequenceMatcher

from ctor.belief_propag import PGM, LabeledArray, Messages, NoSolutionError


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
        # Kept your original semantics
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

    def add_viewpoint_realization_old(self, i, sequence_index, vp):
        new_address = (sequence_index, i)
        self.viewpoints_realizations[vp].append(new_address)

    def add_viewpoint_realization_new(self, i, sequence_index, vp):
        new_address = (sequence_index, i)
        if self.is_starting_address(new_address) or self.is_ending_address(new_address):
            self.viewpoints_realizations[vp].append(new_address)
            return
        new_note = self.get_input_object(new_address)
        for real in self.viewpoints_realizations[vp]:
            real_note = self.get_input_object(real)
            if real_note.is_similar_realization(new_note):
                return
        self.viewpoints_realizations[vp].append(new_address)

    add_viewpoint_realization = add_viewpoint_realization_old

    # ------------------ matrices ------------------

    def get_first_order_matrix_old(self):
        """
        Legacy-style matrix from FULL counts on order-1 contexts, using the unified dictionary.
        """
        keys = self.all_unique_viewpoints
        n = len(keys)
        result = np.zeros((n, n), dtype=float)
        now = self.global_step if self.decay_freeze_at is None else self.decay_freeze_at
        for i_vp, vp in enumerate(keys):
            mc = self.ctx_to_continuations.get((vp,), None)
            if not mc:
                continue
            row_counts = mc.weights(now, "full")
            for j, c in row_counts.items():
                result[i_vp, j] = c
            s = result[i_vp].sum()
            if s > 0:
                result[i_vp] /= s
        return result

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

    # ------------------ viewpoints & sampling ------------------

    def get_viewpoint(self, real_object):
        if self.viewpoint_lambda is None:
            return real_object
        return self.viewpoint_lambda(real_object)

    def get_realizations_for_vp(self, vp):
        return self.viewpoints_realizations[vp]

    def random_starting_note(self):
        # unchanged (uses your previous address scheme)
        starting_vp = (-1, 0)
        starting_conts = self.get_realizations_for_vp(starting_vp)
        start = random.choice(starting_conts)
        return start

    def sample_sequence_that_ends(self, start_vp, length=50):
        pgm = self.build_bp_graph(length)
        pgm.set_value('x1', self.index_of_vp(start_vp))
        pgm.set_value('x' + str(length + 2), self.index_of_vp(self.end_padding))
        try:
            vp_seq = self.sample_vp_sequence_with_bp(start_vp, length, pgm)
        except NoSolutionError:
            return None
        return vp_seq

    def sample_sequence(self, length, constraints=None):
        if len(self.input_sequences) == 0:
            return None
        pgm = self.build_bp_graph(length)
        start_vp = None
        if constraints is not None:
            for ct_pos, ct_vp in constraints.items():
                var_name = "x" + str(ct_pos + 1)
                pgm.set_value(var_name, self.index_of_vp(ct_vp))
            if 0 in constraints:
                start_vp = constraints[0]
        try:
            vp_seq = self.sample_vp_sequence_with_bp(length, start_vp, pgm)
        except NoSolutionError:
            print("too many constraints?")
            return None
        return vp_seq

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

    @staticmethod
    def is_ok(marginal):
        for x in marginal:
            if np.isnan(x):
                return False
        return True

    def sample_vp_sequence_with_bp(self, length, start_vp, pgm):
        if length < 0:
            print("impossible")
        if start_vp is not None:
            current_seq = [start_vp]
        else:
            try:
                marginal_1 = Messages().marginal(pgm.variable_from_name('x1'))
                vp = self.random_vp_with_probs(marginal_1)
                current_seq = [vp]
                pgm.set_value('x1', self.index_of_vp(current_seq[0]))
            except NoSolutionError:
                return None
        # generate the rest of the sequence
        first_order_matrix = self.get_first_order_matrix()
        for i in range(length - 1):
            pgm_variable = pgm.variable_from_name('x' + str(i + 2))
            try:
                marginal_i = Messages().marginal(pgm_variable)
            except NoSolutionError:
                return None
            markov_proba = first_order_matrix[self.index_of_vp(current_seq[-1])]
            product_proba = marginal_i * markov_proba
            cont = self.get_continuation_with_bp(current_seq, product_proba)
            if cont == -1:
                cont = self.random_initial_vp()
            current_seq.append(cont)
            pgm.set_value('x' + str(i + 2), self.index_of_vp(cont))
        return current_seq

    def sample_vp_sequence(self, start_vp, length, end_vp):
        current_seq = [start_vp]
        if length >= 0:
            for _ in range(length):
                cont = self.get_continuation(current_seq)
                if cont == -1:
                    cont = self.random_initial_vp()
                current_seq.append(cont)
            return current_seq
        while True:
            cont = self.get_continuation(current_seq)
            if cont == -1:
                cont = self.random_initial_vp()
            if cont == end_vp:
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
        return -1

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
        return -1

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
