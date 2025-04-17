from collections import Counter

import numpy as np
import random
from difflib import SequenceMatcher

from ctor.belief_propag import PGM, LabeledArray, Messages, NoSolutionError


class _Start_vp:
    def __init__(self):
        pass


class _End_vp:
    def __init__(self):
        pass


class Variable_order_Markov:
    def __init__(self, sequence_of_stuff, vp_lambda, kmax=5):
        # the input sequences of realizations
        self.viewpoint_lambda = vp_lambda
        self.start_padding = _Start_vp()
        self.end_padding = _End_vp()
        self.kmax = kmax
        self.input_sequences = []
        self.all_unique_viewpoints = []
        self.viewpoints_realizations = {}
        self.prefixes_to_continuations = np.empty(self.kmax, dtype=object)
        for k in range(self.kmax):
            self.prefixes_to_continuations[k] = {}
        if sequence_of_stuff is not None:
            self.learn_sequence(sequence_of_stuff)

    def clear_memory(self):
        self.input_sequences = []
        self.all_unique_viewpoints = []
        self.viewpoints_realizations = {}
        self.prefixes_to_continuations = np.empty(self.kmax, dtype=object)
        for k in range(self.kmax):
            self.prefixes_to_continuations[k] = {}

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
        self.input_sequences.append(sequence_of_stuff)
        self.build_vo_markov_model(sequence_of_stuff)

    def get_input_object(self, obj_address):
        # note_address is a tuple (melody index, index in melody)
        return self.input_sequences[obj_address[0]][obj_address[1]]

    @staticmethod
    def is_starting_address(note_address):
        return note_address[1] == 1

    def is_ending_address(self, note_address):
        return note_address[1] == len(self.input_sequences[note_address[0]]) - 2

    def is_end_padding(self, vp):
        return vp == self.end_padding

    def voc_size(self):
        # the number of unique viewpoints, including Start and End viewpoints
        return len(self.all_unique_viewpoints)

    def random_initial_vp(self):
        # returns a random initial vp, which are continuations of start paddings
        all_initial_vps = self.prefixes_to_continuations[0][tuple([self.start_padding])]
        return random.choice(all_initial_vps)

    def random_vp_with_probs(self, probs):
        idx = np.random.choice(len(probs), p=probs)
        return self.get_all_unique_viewpoints()[idx]

    def get_all_unique_viewpoints(self):
        return self.all_unique_viewpoints

    def get_all_unique_viewpoints_except_paddings(self):
        vps = self.all_unique_viewpoints[:]
        vps.remove(self.start_padding)
        vps.remove(self.end_padding)
        return vps

    def index_of_vp(self, vp):
        return self.all_unique_viewpoints.index(vp)

    def build_vo_markov_model(self, real_sequence):
        """Builds a variable-order Markov model for max K order
        accumulates with existing model"""
        # builds the vp sequence with extra start and end padding vps
        vp_sequence = [self.start_padding] + [self.get_viewpoint(obj) for obj in real_sequence] + [self.end_padding]
        # adds unique viewpoints if any
        for vp in vp_sequence:
            if vp not in self.all_unique_viewpoints:
                self.all_unique_viewpoints.append(vp)
        # add the realization to the viewpoint's realizations
        sequence_index = len(self.input_sequences) - 1
        for i, vp in enumerate(vp_sequence[1:-1]):
            if vp not in self.viewpoints_realizations:
                self.viewpoints_realizations[vp] = []
            self.add_viewpoint_realization(i, sequence_index, vp)
        # populate the prefixes_to_continuations with vp contexts to vps
        for k in range(self.kmax):
            prefixes_to_cont_k = self.prefixes_to_continuations[k]
            for i in range(len(vp_sequence) - k):
                if i < k + 1:
                    continue
                current_ctx = tuple(vp_sequence[i - k - 1: i])
                if current_ctx not in prefixes_to_cont_k:
                    prefixes_to_cont_k[current_ctx] = []
                prefixes_to_cont_k[current_ctx].append(vp_sequence[i])
            self.prefixes_to_continuations[k] = prefixes_to_cont_k
        # special case for the endVp, which has no continuation, but should be in the list for consistency
        end_tuple = tuple([self.end_padding])
        if end_tuple not in self.prefixes_to_continuations[0]:
            # ends goes to end
            self.prefixes_to_continuations[0][end_tuple] = [self.end_padding]

    # returns the priors for all viewpoints (except start and end)
    def get_priors(self):
        # there is no start and end vps in this list
        key_counts = {key: len(continuations) for key, continuations in self.viewpoints_realizations.items()}
        total_count = sum(key_counts.values())
        priors = {key: count / total_count for key, count in key_counts.items()}
        # Step 4: Convert to a sorted vector (optional)
        sorted_keys = self.get_all_unique_viewpoints_except_paddings()
        # Ensure consistent ordering
        probability_vector = np.array([priors[key] for key in sorted_keys])
        return probability_vector

    def sample_zero_order(self, k):
        priors = self.get_priors()
        return random.choices(self.get_all_unique_viewpoints_except_paddings(), weights=priors, k=k)

    def add_viewpoint_realization_old(self, i, sequence_index, vp):
        # attention! vp sequence has extra start_vp, so i should be decreased by 1!
        new_address = tuple([sequence_index, i])
        self.viewpoints_realizations[vp].append(new_address)

    def add_viewpoint_realization_new(self, i, sequence_index, vp):
        # adds only if different from existing ones, to avoid inflation in case of monotonous pieces
        new_address = tuple([sequence_index, i])
        if self.is_starting_address(new_address) or self.is_ending_address(new_address):
            # starting address are added, cause useful at rendering time
            self.viewpoints_realizations[vp].append(new_address)
            return
        new_note = self.get_input_object(new_address)
        for real in self.viewpoints_realizations[vp]:
            real_note = self.get_input_object(real)
            if real_note.is_similar_realization(new_note):
                return
        self.viewpoints_realizations[vp].append(new_address)

    add_viewpoint_realization = add_viewpoint_realization_old

    def get_first_order_matrix(self):
        # returns the matrix for first order Markov transitions
        # all states. This includes start and end padding states
        keys = self.get_all_unique_viewpoints()
        result = np.zeros((len(keys), len(keys)))
        k0 = self.prefixes_to_continuations[0]
        for i_vp, vp in enumerate(keys):
            conts = k0[tuple([vp])]
            occurrences = Counter(conts)
            for vp2 in occurrences:
                i_vp2 = keys.index(vp2)
                result[i_vp, i_vp2] = occurrences[vp2]
            result[i_vp] /= result[i_vp].sum()
        return result

    def get_viewpoint(self, real_object):
        if self.viewpoint_lambda is None:
            return real_object
        return self.viewpoint_lambda(real_object)

    def get_realizations_for_vp(self, vp):
        return self.viewpoints_realizations[vp]

    def random_starting_note(self):
        starting_vp = (-1, 0)
        starting_conts = self.get_realizations_for_vp(starting_vp)
        start = random.choice(starting_conts)
        return start

    def sample_sequence_that_ends(self, start_vp, length=50):
        # if length is negative, stops when reaching the provided end_viewpoint
        # if nb_sequences is positive, stops after nb_sequences occurrences of the end_vp

        pgm = self.build_bp_graph(length)
        # sets constraints on start and end
        pgm.set_value('x1', self.index_of_vp(start_vp))
        pgm.set_value('x' + str(length + 2), self.index_of_vp(self.end_padding))
        # with BP
        try:
            vp_seq = self.sample_vp_sequence_with_bp(start_vp, length, pgm)
        except NoSolutionError:
            return None
        return vp_seq

    def sample_sequence(self, length, constraints=None):
        # if length is negative, stops when reaching the provided end_viewpoint
        # if nb_sequences is positive, stops after nb_sequences occurrences of the end_vp
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
        # assert is_conditional_prob(mat, "x2")
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
        # Generates a new sequence of vps from the Markov model.
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
                # if not self.is_ok(marginal_i):
            except NoSolutionError:
                return None
            # compare with the markov transition matrix
            markov_proba = first_order_matrix[self.index_of_vp(current_seq[-1])]
            product_proba = marginal_i * markov_proba
            cont = self.get_continuation_with_bp(current_seq, product_proba)
            if cont == -1:
                print("should not be here,there is always a continuation with BP")
                cont = self.random_initial_vp()
            current_seq.append(cont)
            pgm.set_value('x' + str(i + 2), self.index_of_vp(cont))
        return current_seq

    def sample_vp_sequence(self, start_vp, length, end_vp):
        # Generates a new sequence of vps from the Markov model.
        current_seq = [start_vp]
        if length >= 0:
            # generate fixed length sequence
            for _ in range(length):
                cont = self.get_continuation(current_seq)
                if cont == -1:
                    print("restarting from scratch")
                    cont = self.random_initial_vp()
                current_seq.append(cont)
            return current_seq
        while True:
            cont = self.get_continuation(current_seq)
            if cont == -1:
                print("restarting from scratch")
                cont = self.random_initial_vp()
            if cont == end_vp:
                print("found the end")
                if cont != self.end_padding:
                    current_seq.append(cont)
                return current_seq
            current_seq.append(cont)

    def get_continuation(self, current_seq):
        vp_to_skip = None
        for k in range(self.kmax, 0, -1):
            if k > len(current_seq):
                continue
            continuations_dict = self.prefixes_to_continuations[k - 1]
            viewpoint_ctx = tuple(current_seq[-k:])
            if viewpoint_ctx in continuations_dict:
                all_cont_vps = continuations_dict[viewpoint_ctx]
                # considers the number of different viewpoints, not the number of continuations as they are repeated
                if len(set(all_cont_vps)) == 1 and k > 1:
                    # proba to skip is proportional to order
                    if random.random() > (1 / (k + 1)):
                        # print(f"skipping continuation for {k=}")
                        vp_to_skip = all_cont_vps[0]
                        continue
                    else:
                        vp_to_skip = None
                        # print(f"not skipping singleton continuation for {k=}")
                if vp_to_skip is not None and k > 1:
                    conts_to_use = [c for c in all_cont_vps if c != vp_to_skip]
                else:
                    conts_to_use = all_cont_vps
                next_continuation = random.choice(conts_to_use)
                return next_continuation
        print("no continuation found")
        return -1

    def get_continuation_with_bp(self, current_seq, probs):
        vp_to_skip = None
        for k in range(self.kmax, 0, -1):
            if k > len(current_seq):
                continue
            continuations_dict = self.prefixes_to_continuations[k - 1]
            viewpoint_ctx = tuple(current_seq[-k:])
            if viewpoint_ctx in continuations_dict:
                all_cont_vps = continuations_dict[viewpoint_ctx]
                # filters out the continuations with low probabilities
                all_cont_vps = [vp for vp in all_cont_vps if probs[self.index_of_vp(vp)] > 0]
                if len(all_cont_vps) == 0:
                    # print("continuations removed by bp")
                    continue
                # considers the number of different viewpoints, not the number of continuations as they are repeated
                if len(set(all_cont_vps)) == 1 and k > 1:
                    # proba to skip is proportional to order
                    if random.random() > (1 / (k + 1)):
                        # print(f"skipping continuation for {k=}")
                        vp_to_skip = all_cont_vps[0]
                        continue
                    else:
                        vp_to_skip = None
                        # print(f"not skipping singleton continuation for {k=}")
                if vp_to_skip is not None and k > 1:
                    conts_to_use = [c for c in all_cont_vps if c != vp_to_skip]
                else:
                    conts_to_use = all_cont_vps
                next_continuation = random.choice(conts_to_use)
                # print(f"{k=}")
                return next_continuation
        print("no continuation found")
        return -1

    def show_conts_structure(self):
        for k in range(self.kmax):
            print(
                f"size of contexts of size {k + 1}: {len(self.prefixes_to_continuations[k])}"
            )
        # looks at the sparsity of the matrix
        order1 = self.prefixes_to_continuations[0]
        voc_size = self.voc_size()
        min_size = voc_size
        max_size = 0
        for voc in order1.keys():
            conts_size = len(set(order1[voc]))
            if conts_size > max_size:
                max_size = conts_size
            if conts_size < min_size:
                min_size = conts_size
        print(f"voc size: {voc_size}")
        print(f"min order 1 size: {min_size}, max: {max_size}")
        total = 0
        for k in self.viewpoints_realizations:
            total += len(self.viewpoints_realizations[k])
        print(f"average nb of vp realizations: {total / voc_size}")
