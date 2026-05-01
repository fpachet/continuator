"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

import numpy as np


class VariableDomainSequenceOptimizer:
    """
    A class for solving sequence assignment problems with variable domains:

    We have positions i = 0..n-1, each with its own domain[i].
    We want to minimize:
        sum_{i=0}^{n-1} unary_cost(i, x_i)
      + sum_{i=0}^{n-2} binary_cost(i, x_i, i+1, x_{i+1})

    using dynamic programming, supporting different domain sizes per position.
    """

    def __init__(self, domains, unary_cost, binary_cost):
        """
        Parameters
        ----------
        domains : list of lists
            domains[i] is the list of allowable labels for position i.
            E.g., domains[0] = [0,1,2], domains[1] = ['A','B'], etc.
        unary_cost : function (i, x) -> float
            A function that gives the cost of assigning value x at position i.
        binary_cost : function (i, x, i+1, y) -> float
            A function that gives the cost of assigning x at position i and y at position i+1.
        """
        # n is the total number of positions
        self.n = len(domains)
        # domains[i] is a list of valid labels for position i
        self.domains = domains
        # cost functions
        self.unary_cost_func = unary_cost
        self.binary_cost_func = binary_cost

        # Precompute unary arrays: U[i], shape (|D_i|,)
        self.U = self._compute_unary_arrays()

        # Precompute binary arrays: B[i], shape (|D_i|, |D_{i+1}|)
        if self.n > 1:
            self.B = self._compute_binary_arrays()
        else:
            self.B = []  # no binary cost if only one position

        # DP tables (filled by fit)
        # dp[i] will be shape (|D_i|,)
        # backpointer[i] will be shape (|D_i|,) storing the chosen index in domain[i+1]
        self.dp = [None] * self.n
        self.backpointer = [None] * self.n

    def _compute_unary_arrays(self):
        """
        For each position i, create a 1D array of shape (|D_i|,)
        where U[i][d] = unary_cost_func(i, domains[i][d]).
        """
        U = []
        for i in range(self.n):
            dom_i = self.domains[i]
            arr = np.zeros(len(dom_i), dtype=np.float64)
            for d, label in enumerate(dom_i):
                arr[d] = self.unary_cost_func(i, label)
            U.append(arr)
        return U

    def _compute_binary_arrays(self):
        """
        For each i in [0..n-2], create a 2D array B[i] of shape (|D_i|, |D_{i+1}|)
        where B[i][d1, d2] = binary_cost_func(i, domains[i][d1], i+1, domains[i+1][d2]).
        """
        B = []
        for i in range(self.n - 1):
            dom_i = self.domains[i]
            dom_next = self.domains[i + 1]
            mat = np.zeros((len(dom_i), len(dom_next)), dtype=np.float64)
            for d1, label1 in enumerate(dom_i):
                for d2, label2 in enumerate(dom_next):
                    mat[d1, d2] = self.binary_cost_func(i, label1, i + 1, label2)
            B.append(mat)
        return B

    def fit(self):
        """
        Run the dynamic programming to find the minimum total cost and the best assignment.

        Returns
        -------
        (min_cost, best_sequence)
          min_cost : float
              The minimal total cost.
          best_sequence : list
              A list of length n with the optimal label for each position.
        """
        # If n == 0, trivial
        if self.n == 0:
            return 0.0, []

        # dp[i], shape (|D_i|,) -> minimal cost from position i onward if x_i = domain[i][d]
        # backpointer[i], shape (|D_i|,) -> best index in domain[i+1] for each d in domain[i].

        # Base case: i = n-1
        self.dp[self.n - 1] = self.U[self.n - 1].copy()
        self.backpointer[self.n - 1] = np.full(len(self.domains[self.n - 1]), -1, dtype=np.int64)

        # Fill backward from i = n-2 down to 0
        for i in range(self.n - 2, -1, -1):
            dom_i_size = len(self.domains[i])
            dom_next_size = len(self.domains[i + 1])

            dp_i = np.zeros(dom_i_size, dtype=np.float64)
            bp_i = np.zeros(dom_i_size, dtype=np.int64)

            # cost_matrix = B[i] + dp[i+1]
            # B[i] is shape (dom_i_size, dom_next_size)
            # dp[i+1] is shape (dom_next_size,)
            # so cost_matrix is shape (dom_i_size, dom_next_size), where
            # cost_matrix[d1, d2] = B[i][d1, d2] + dp[i+1][d2]
            cost_matrix = self.B[i] + self.dp[i + 1]

            # For each d1 in [0..dom_i_size-1], we find the minimal cost over d2
            # min_costs[d1] = min_{d2} [ cost_matrix[d1, d2] ]
            # best_next[d1] = argmin_{d2} [ cost_matrix[d1, d2] ]
            min_costs = np.min(cost_matrix, axis=1)
            best_next = np.argmin(cost_matrix, axis=1)

            dp_i[:] = self.U[i] + min_costs
            bp_i[:] = best_next

            self.dp[i] = dp_i
            self.backpointer[i] = bp_i

        # Find the best start label at i=0
        min_cost = np.min(self.dp[0])
        best_start = np.argmin(self.dp[0])

        # Reconstruct solution
        best_sequence = [None] * self.n
        best_sequence[0] = self.domains[0][best_start]

        prev_index = best_start
        for i in range(0, self.n - 1):
            next_index = self.backpointer[i][prev_index]
            best_sequence[i + 1] = self.domains[i + 1][next_index]
            prev_index = next_index

        return min_cost, best_sequence


# ---------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Suppose we have 4 positions, each with a different domain of labels:
    domains = [
        [0, 1],  # position 0
        [0, 1, 2],  # position 1
        ['A', 'B'],  # position 2
        [10, 20, 30]  # position 3
    ]


    # A simple unary cost function that depends on i and x
    def unary_cost(i, x):
        # e.g., cost is i * int(x != 0) just as a silly example
        # for non-integer x, we'll treat 'A'/'B' or whatever carefully
        return 1.0 if x != 0 else 0.0


    # A simple binary cost function
    def binary_cost(i, x, j, y):
        # For demonstration, let's say cost = 1 if x == y, else 0
        return float(x == y)


    optimizer = VariableDomainSequenceOptimizer(domains, unary_cost, binary_cost)
    cost, best_seq = optimizer.fit()

    print("Minimal cost:", cost)
    print("Best sequence:", best_seq)
