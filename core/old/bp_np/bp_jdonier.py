import random

import numpy as np
from tqdm import tqdm

# from .sample.sample import get_samples, sample_from


def one_hot(n, size):
    arr = np.zeros(size)
    arr[n] = 1
    return arr


class FlatGraph:
    """
    A flat graph object that will hold the unary and binary factors and
    implement belief propagation.

    The graph should be used as follows:

    1) Create the graph with `num_nodes` nodes,
    2) Add unary factors for all `num_nodes` node with `add_unary_factor`,
    3) Add binary factors for all `num_nodes - 1` pair of neighbours with
    `add_binary_factor`
    4) Sample with `sample` (Cython version) or `sample_python` (Python version)

    """

    def __init__(self, num_nodes, domain_size):

        assert num_nodes > 0

        self.num_nodes = num_nodes
        self.domain_size = domain_size

        self.unary_factors = np.zeros((self.num_nodes, domain_size), dtype=np.float32)
        self.binary_factors = np.zeros(
            (self.num_nodes - 1, domain_size, domain_size), dtype=np.float32
        )
        self.messages_from_right = np.zeros(
            (self.num_nodes, domain_size), dtype=np.float32
        )

    def add_unary_factor(self, i, values):

        assert type(values) == np.ndarray
        assert values.shape == (self.domain_size,)

        self.unary_factors[i] = values

    def add_binary_factor(self, i, j, values):
        assert j == i + 1
        assert type(values) == np.ndarray
        assert values.shape == (self.domain_size, self.domain_size)

        self.binary_factors[i] = values

    def validate(self):
        assert self.num_nodes == len(self.unary_factors) == len(self.binary_factors) + 1

    def backward_pass(self):

        current_message = np.ones(self.domain_size, dtype=np.float32)
        self.messages_from_right[self.num_nodes - 1] = current_message

        for i in range(1, self.num_nodes)[::-1]:
            new_message = np.matmul(
                self.binary_factors[i - 1], self.unary_factors[i] * current_message
            )

            # Normalize
            current_message = new_message / np.sum(new_message)

            self.messages_from_right[i - 1] = current_message

    def sample(self, num_samples=1, return_probas=False, **kwargs):
        self.backward_pass()

        # return get_samples(num_samples,
        #                    self.unary_factors.astype(np.float64),
        #                    self.binary_factors.astype(np.float64),
        #                    self.messages_from_right.astype(np.float64),
        #                    return_probas)

        return self.sample_python(
            num_samples,
            self.unary_factors.astype(np.float64),
            self.binary_factors.astype(np.float64),
            self.messages_from_right.astype(np.float64),
            return_probas,
        )

    def sample_python(
        self, num_samples=1, return_probas=False, print_progress=False, **kwargs
    ):

        self.backward_pass()

        iterable = range(num_samples)
        if print_progress:
            iterable = tqdm(iterable)

        samples_list = np.zeros((num_samples, self.num_nodes), dtype=np.int32)

        if return_probas:
            probas_unary = np.ones((num_samples, self.num_nodes), dtype=np.float32)
            probas_binary = np.ones((num_samples, self.num_nodes), dtype=np.float32)

        for n in iterable:

            p = self.messages_from_right[0] * self.unary_factors[0]
            p /= np.sum(p)

            idx = np.argmax(
                np.cumsum(p) > random.random()
            )  # sample_from(p, random.random())
            samples_list[n, 0] = idx

            if return_probas:
                probas_unary[n, 0] = self.unary_factors[0, idx]
                previous_idx = idx

            # Sample from the nodes in time forward order and propagate any
            # skip messages
            for i in range(1, self.num_nodes):

                # Compute proba
                p = (
                    self.binary_factors[i - 1, idx]
                    * self.messages_from_right[i]
                    * self.unary_factors[i]
                )

                # Sample and set a value for the current node
                idx = sample_from(p, random.random())
                samples_list[n, i] = idx

                if return_probas:
                    probas_unary[n, i] = self.unary_factors[i, idx]
                    probas_binary[n, i] = self.binary_factors[i - 1, previous_idx, idx]
                    previous_idx = idx

        if return_probas:
            return samples_list, probas_unary, probas_binary

        return samples_list

    def optimal_sequence(self, return_probas=False):
        proba = self.unary_factors[0]
        traceback = []

        for i in range(1, self.num_nodes):
            proba = (
                self.binary_factors[i - 1]
                * proba[:, None]
                * self.unary_factors[i][None, :]
            )

            # Normalizes the probability matrix, to prevent probabilities from
            # vanishing to 0.0
            proba = proba / np.sum(proba)

            best_previous_node = np.argmax(proba, axis=0)
            proba = np.max(proba, axis=0)
            traceback.append(best_previous_node)

        indices = [np.argmax(proba)]
        for best_previous_node in traceback[::-1]:
            indices.append(best_previous_node[indices[-1]])

        indices = indices[::-1]

        if return_probas:
            return (
                np.array(indices),
                np.array([self.unary_factors[i, idx] for i, idx in enumerate(indices)]),
                np.array(
                    [1]
                    + [
                        self.binary_factors[i, idx1, idx2]
                        for i, (idx1, idx2) in enumerate(zip(indices[:-1], indices[1:]))
                    ]
                ),
            )

        return np.array(indices)


class Node:
    """
    Node objects which records its position on the time axis (starting at 0),
    as well as the pointers to / messages from:

    - the immediately previous node on the time axis,
    - the immediately previous node on the time axis,
    - the nodes on its left to which it has been additionally linked,
    - the nodes on its right to which it has been additionally linked,

    If a value is set (subsequent to sampling), it supersedes any messages
    from neighbours.

    Parameters
    ----------
    position : int
        the position of the node in the graph
    domain_size: int
        the number of states the node variable can take

    """

    def __init__(self, position, domain_size):
        self.position = position

        self.previous = None
        self.next = None
        self.skip_left = []
        self.skip_right = []

        self.message_from_previous = None
        self.message_from_next = None
        self.messages_from_skip_left = {}
        self.messages_from_skip_right = {}

        self.value = None

    def __repr__(self):
        return "N%d" % self.position


class Graph:
    """
    The graph object that will hold the nodes and implement belief propagation.

    The graph should be used as follows:

    1) Create the graph with `num_nodes` nodes,
    2) Add unary factors for all `num_nodes` node with `add_unary_factor`,
    3) Add binary factors for all `num_nodes - 1` pair of neighbours with
    `add_binary_factor`,
    4) Optionally, add link between non-adjacent nodes with `add_binary_factor`,

    """

    def __init__(self, num_nodes, domain_size):

        self.num_nodes = num_nodes
        self.domain_size = domain_size

        self.nodes = []
        last_node = None
        for i in range(num_nodes):

            # Create node
            current_node = Node(i, domain_size)

            self.nodes.append(current_node)

            # Link neighbours
            if last_node is not None:
                current_node.previous = last_node
                last_node.next = current_node

            last_node = current_node

        self.factors = {}

    def __len__(self):
        """Equal to the number of nodes in the graph"""
        return self.num_nodes

    def __iter__(self):
        """Iterate over nodes in temporal order"""
        for node in self.nodes:
            yield node

    def __repr__(self):
        """A representation with node names and skip links"""
        line = ""
        nodes_pos = []
        for node in self.nodes:
            nodes_pos.append(len(line))
            line += " " + node.__repr__()

        depth = 1
        for node in self.nodes:
            for skip_right in node.skip_right:
                line += "\n"
                left_pos = nodes_pos[node.position] + 1
                delta_pos = nodes_pos[skip_right.position] + 1 - left_pos - 1
                line += " " * left_pos + "|" + "_" * delta_pos + "|"
                depth += 1

        return line

    def clear(self):
        """Clear nodes values and messages, for running a new sampling."""
        for node in self:
            node.value = None
            node.message_from_previous = None
            node.message_from_next = None
            node.messages_from_skip_left = {}
            node.messages_from_skip_right = {}

    def validate(self):

        required_keys = []

        for node in self:
            assert node.position in self.factors
            required_keys.append(node.position)

        for node in self.nodes[:-1]:
            assert (node.position, node.position + 1) in self.factors
            required_keys.append((node.position, node.position + 1))

        for node in self:
            for skip_left in node.skip_left:
                assert (skip_left.position, node.position) in self.factors
                required_keys.append((skip_left.position, node.position))

        for node in self:
            for skip_right in node.skip_right:
                assert (node.position, skip_right.position) in self.factors
                required_keys.append((node.position, skip_right.position))

        for key in self.factors.keys():
            assert key in required_keys

    def check_cleared(self):
        """Clear nodes values and messages, for running a new sampling."""
        for node in self:
            assert node.value is None
            assert node.message_from_previous is None
            assert node.message_from_next is None
            assert len(node.messages_from_skip_left) == 0
            assert len(node.messages_from_skip_right) == 0

    def add_unary_factor(self, i, values):
        """
        Add unary factor for a node

        Parameters
        ----------
        i: int
            position of the node on the graph
        values: np.ndarray
            1-D array containing the value of the factor the states.
            Must have length `self.domain_size`.
        """
        assert 0 <= i < len(self)
        assert np.round(i) == i

        i = int(i)

        assert type(values) == np.ndarray
        assert values.shape == (self.domain_size,)

        self.factors[i] = values.astype(float)

    def add_binary_factor(self, i, j, values):
        """
        Add binary factor for a node

        Parameters
        ----------
        i: int
            position of the first node on the graph
        j: int
            position of the second node on the graph
        values: np.ndarray
            2-D array containing the value of the factor the joint states.
            Must have shape `(self.domain_size, self.domain_size)`.
        """
        assert 0 <= i < len(self)
        assert 0 <= j < len(self)

        assert np.round(i) == i
        assert np.round(j) == j

        i = int(i)
        j = int(j)

        assert type(values) == np.ndarray
        assert values.shape == (self.domain_size, self.domain_size)

        min_idx = min(i, j)
        max_idx = max(i, j)

        # Add skip links for both nodes
        if max_idx > min_idx + 1:
            assert self.nodes[max_idx] not in self.nodes[min_idx].skip_right
            assert self.nodes[min_idx] not in self.nodes[max_idx].skip_left
            self.nodes[min_idx].skip_right.append(self.nodes[max_idx])
            self.nodes[max_idx].skip_left.append(self.nodes[min_idx])

        assert (i, j) not in self.factors, "Factor (%d, %d) already set" % (i, j)
        self.factors[(i, j)] = values.astype(float)

    def run(self, n_iter=1, delta_skip=0.8, delta_next=0.95, from_node=0, to_node=None):
        """
        Run a first causal forward pass then a backward pass.
        Additional forward and backward passes can be run if `n_iter` is >1.

        Parameters
        ----------
        n_iter: int
            the number of additional round trips to make
        delta_skip: float
            the exponent in [0, 1] for dampening skip messages and avoiding
            divergence in loopy graphs
        delta_next: float
            the exponent in [0, 1] for dampening neighbours messages and
            avoiding divergence in loopy graphs
        from_node: int
            the leftmost node for this run of the graph
        to_node: int or None
            the rightmost node for this run of the graph
        """
        assert n_iter >= 1

        self.forward_pass(
            reverse=False,
            delta_skip=delta_skip,
            delta_next=delta_next,
            from_node=from_node,
            to_node=to_node,
        )
        self.backward_pass(
            delta_skip=delta_skip,
            delta_next=delta_next,
            from_node=from_node,
            to_node=to_node,
        )

        for _ in range(n_iter - 1):
            self.forward_pass(
                delta_skip=delta_skip,
                delta_next=delta_next,
                from_node=from_node,
                to_node=to_node,
            )
            self.backward_pass(
                delta_skip=delta_skip,
                delta_next=delta_next,
                from_node=from_node,
                to_node=to_node,
            )

    def forward_pass(
        self, reverse=True, delta_skip=0.8, delta_next=0.95, from_node=0, to_node=None
    ):
        """
        Forward pass.

        Parameters
        ----------
        reverse: bool
            whether to include skip messages from the right (usually only
            False for the first pass)
        delta_skip: float
            the exponent in [0, 1] for dampening skip messages and avoiding
            divergence in loopy graphs
        delta_next: float
            the exponent in [0, 1] for dampening neighbours messages and
            avoiding divergence in loopy graphs
        from_node: int
            the leftmost node for this run of the graph
        to_node: int or None
            the rightmost node for this run of the graph
        """
        for node in self.nodes[from_node:to_node]:

            if node.next is None:
                return

            if node.value is None:

                # Compute the universal part of the message that the current
                # node will
                # send to all right neighbours and skips, starting from the
                # unary factor
                incoming_message = np.array(self.factors[node.position])

                # Message from previous node
                if node.message_from_previous is not None:
                    incoming_message *= node.message_from_previous**delta_next

                # Messages from skip left nodes
                for skip_left in node.skip_left:
                    incoming_message *= (
                        node.messages_from_skip_left[skip_left] ** delta_skip
                    )

                if reverse:
                    # Messages from skip right nodes
                    for skip_right in node.skip_right:
                        incoming_message *= (
                            node.messages_from_skip_right[skip_right] ** delta_skip
                        )

                # Normalize
                incoming_message /= np.sum(incoming_message)

                node.next.message_from_previous = np.matmul(
                    incoming_message, self.factors[(node.position, node.position + 1)]
                )

                # Message to skip right nodes
                for skip_right in node.skip_right:

                    # Message from next node
                    if node.message_from_next is not None:
                        incoming_message_from_next = node.message_from_next**delta_next
                    else:
                        incoming_message_from_next = 1

                    if reverse:
                        message_skip = incoming_message / (
                            1e-15
                            + node.messages_from_skip_right[skip_right] ** delta_skip
                        )
                    else:
                        message_skip = incoming_message

                    message_skip *= incoming_message_from_next

                    skip_right.messages_from_skip_left[node] = np.matmul(
                        message_skip, self.factors[(node.position, skip_right.position)]
                    )

            else:
                node.next.message_from_previous = self.factors[
                    (node.position, node.position + 1)
                ][node.value]

                # Message to skip right nodes
                for skip_right in node.skip_right:
                    skip_right.messages_from_skip_left[node] = self.factors[
                        (node.position, skip_right.position)
                    ][node.value]

    def backward_pass(
        self, reverse=True, delta_skip=0.8, delta_next=0.95, from_node=0, to_node=None
    ):
        """
        Backward pass.

        Parameters
        ----------
        reverse: bool
            whether to include skip messages from the left (usually always True)
        delta_skip: float
            the exponent in [0, 1] for dampening skip messages and avoiding
            divergence in loopy graphs
        delta_next: float
            the exponent in [0, 1] for dampening neighbours messages and
            avoiding divergence in loopy graphs
        from_node: int
            the leftmost node for this run of the graph
        to_node: int or None
            the rightmost node for this run of the graph
        """
        to_node = to_node if to_node is None else to_node + 1
        for node in self.nodes[from_node + 1 : to_node][::-1]:

            if node.previous is None:
                return

            if node.value is None:
                # Compute the universal part of the message that the current
                # node will
                # send to all left neighbours and skips, starting from the
                # unary factor
                incoming_message = np.array(self.factors[node.position])

                # Message from next node
                if node.message_from_next is not None:
                    incoming_message *= node.message_from_next**delta_next

                if reverse:
                    # Messages from skip left nodes
                    for skip_left in node.skip_left:
                        incoming_message *= (
                            node.messages_from_skip_left[skip_left] ** delta_skip
                        )

                # Messages from skip right nodes
                for skip_right in node.skip_right:
                    incoming_message *= (
                        node.messages_from_skip_right[skip_right] ** delta_skip
                    )

                # Normalize
                incoming_message /= np.sum(incoming_message)

                # Message to previous node
                node.previous.message_from_next = np.matmul(
                    self.factors[(node.position - 1, node.position)], incoming_message
                )

                # Message to skip left nodes
                for skip_left in node.skip_left:

                    # Message from previous node
                    if node.message_from_previous is not None:
                        incoming_message_from_previous = (
                            node.message_from_previous**delta_next
                        )
                    else:
                        incoming_message_from_previous = 1

                    if reverse:
                        message_skip = incoming_message / (
                            1e-15
                            + node.messages_from_skip_left[skip_left] ** delta_skip
                        )
                    else:
                        message_skip = incoming_message

                    message_skip *= incoming_message_from_previous

                    skip_left.messages_from_skip_right[node] = np.matmul(
                        self.factors[(skip_left.position, node.position)], message_skip
                    )

            else:
                # Message to previous node
                node.previous.message_from_next = self.factors[
                    (node.position - 1, node.position)
                ][:, node.value]

                # Message to skip left nodes
                for skip_left in node.skip_left:
                    skip_left.messages_from_skip_right[node] = self.factors[
                        (skip_left.position, node.position)
                    ][:, node.value]

    def sample(
        self,
        num_samples=1,
        return_probas=False,
        n_iter=1,
        delta_skip=0.8,
        delta_next=0.95,
        print_progress=False,
    ):
        """
        Sample from the graph.

        Parameters
        ----------
        num_samples: int
            number of samples to draw
        n_iter: int
            the number of additional round trips to make
        delta_skip: float
            the exponent in [0, 1] for dampening skip messages and avoiding
            divergence in loopy graphs
        delta_next: float
            the exponent in [0, 1] for dampening neighbours messages and
            avoiding divergence in loopy graphs
        print_progress: bool
            whether to use a progress bar to be used for the iteration

        Returns
        -------
        samples: np.ndarray
            sampled values for all the node, in time forward order
        """

        iterable = range(num_samples)
        if print_progress:
            iterable = tqdm(iterable)

        samples_list = np.zeros((num_samples, self.num_nodes), dtype=np.int32)

        if return_probas:
            probas_unary = np.ones((num_samples, self.num_nodes), dtype=np.float32)
            probas_binary = np.ones((num_samples, self.num_nodes), dtype=np.float32)

        for n in iterable:

            # Clear any values and messages from a previous sampling
            self.clear()

            # Initialize the messages
            self.run(n_iter=n_iter, delta_skip=delta_skip, delta_next=delta_next)

            # Sample from the nodes in time forward order and propagate any
            # skip messages
            for node in self.nodes:

                # Sample and set a value for the current node
                p = self.proba(node.position)
                sample = np.argmax(np.cumsum(p) > random.random())
                node.value = sample

                samples_list[n, node.position] = sample

                if return_probas:
                    probas_unary[n, node.position] = self.factors[node.position][sample]

                    if node.position > 0:
                        probas_binary[n, node.position] = self.factors[
                            (node.position - 1, node.position)
                        ][previous_sample, sample]

                    previous_sample = sample

                # Default is to pass the message to its right neighbour only
                run_from = node.position
                run_up_to = node.position + 1

                # If the node has a skip right link, then run a full
                # propagation from current position
                if len(node.skip_right) > 0:
                    run_from = node.position
                    run_up_to = None

                # Run the propagation for the scope defined above
                self.run(
                    n_iter=n_iter,
                    delta_skip=delta_skip,
                    delta_next=delta_next,
                    from_node=run_from,
                    to_node=run_up_to,
                )

        if return_probas:
            return samples_list, probas_unary, probas_binary

        return samples_list

    def proba(self, i):
        """
        Probability distribution for a node.

        Parameters
        ----------
        i: int
            position of the node to get the distribution for

        Returns
        -------
        p: np.ndarray
            probability distribution over the values for the node
        """

        node = self.nodes[i]

        # Return value if set by previous sampling
        if node.value is not None:
            return one_hot(node.value, self.domain_size)

        # Otherwise, return marginal distribution.
        # Start from unary factor
        p = np.array(self.factors[i])

        # Then multiply by all incoming messages
        if node.previous is not None:
            p *= node.message_from_previous

        if node.next is not None:
            p *= node.message_from_next

        for _, m in node.messages_from_skip_left.items():
            p *= m

        for _, m in node.messages_from_skip_right.items():
            p *= m

        # assert np.sum(p) > 0, "The probability distribution is degenerate"

        # Return the normalized distribution
        return p / np.sum(p)

    @property
    def samples(self):
        """Get the sampled values of all the nodes after sampling"""
        return np.array([node.value for node in self])

    def optimal_sequence(self, return_probas=False):
        proba = self.factors[0]
        traceback = []

        for node in self.nodes[1:]:
            proba = (
                self.factors[(node.position - 1, node.position)]
                * proba[:, None]
                * self.factors[node.position][None, :]
            )

            # Normalizes the probability matrix, to prevent probabilities from
            # vanishing to 0.0
            proba = proba / np.sum(proba)

            best_previous_node = np.argmax(proba, axis=0)
            proba = np.max(proba, axis=0)
            traceback.append(best_previous_node)

        indices = [np.argmax(proba)]
        for best_previous_node in traceback[::-1]:
            indices.append(best_previous_node[indices[-1]])

        indices = indices[::-1]

        if return_probas:
            return (
                np.array(indices),
                np.array([self.factors[i][idx] for i, idx in enumerate(indices)]),
                np.array(
                    [1]
                    + [
                        self.factors[(i, i + 1)][idx1, idx2]
                        for i, (idx1, idx2) in enumerate(zip(indices[:-1], indices[1:]))
                    ]
                ),
            )

        return np.array(indices)
