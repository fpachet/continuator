import time

import numpy as np


class ChainBP:
    """
    Belief-Propagation (Sum-Product) on a chain X1,...,Xn with:
      - unary_factors[i, k] = phi_i(k) (unnormalized probabilities)
      - trans_matrices[i, j, k] = P(X_{i+1} = k | X_i = j)
    """

    def __init__(self, unary_factors, trans_matrices):
        """
        unary_factors: shape (n, m)
        trans_matrices: shape (n-1, m, m)
        """
        self.n, self.m = unary_factors.shape
        self.unary_factors = unary_factors.astype(
            float
        )  # copy or keep reference as needed
        self.trans_matrices = trans_matrices.astype(float)

        # Placeholders for forward and backward messages
        self.alpha = np.zeros_like(self.unary_factors)  # shape (n, m)
        self.beta = np.zeros_like(self.unary_factors)  # shape (n, m)

    def run_forward_backward(self):
        """Compute alpha, beta arrays (unnormalized) using sum-product on the chain."""
        n, m = self.n, self.m

        # Forward pass: alpha
        # alpha[0, :] = unary_factors[0, :]
        self.alpha[0, :] = self.unary_factors[0, :]
        # Optionally normalize here to improve numerical stability
        norm = self.alpha[0, :].sum()
        if norm > 0:
            self.alpha[0, :] /= norm

        for i in range(1, n):
            # alpha[i] = alpha[i-1] * trans_matrices[i-1], then elementwise multiply by unary_factors[i]
            # shape wise: alpha[i-1,:] is (m,), trans_matrices[i-1] is (m, m)
            tmp = self.alpha[i - 1, :].dot(
                self.trans_matrices[i - 1]
            )  # result shape (m,)
            tmp *= self.unary_factors[i, :]  # elementwise multiply
            # normalize (optional for numerical stability)
            norm = tmp.sum()
            if norm > 0:
                tmp /= norm
            self.alpha[i, :] = tmp

        # Backward pass: beta
        # beta[n-1, :] = 1
        self.beta[n - 1, :] = 1.0
        for i in reversed(range(n - 1)):
            # beta[i, j] = sum_k trans_matrices[i, j, k] * unary_factors[i+1, k] * beta[i+1, k]
            # shape wise: trans_matrices[i] is (m, m), we want a vector (m,) for beta[i].
            # We can do a matrix multiplication if we multiply unary_factors[i+1,:]*beta[i+1,:] first.
            tmp = self.unary_factors[i + 1, :] * self.beta[i + 1, :]
            # shape of tmp is (m,)
            # multiply: trans_matrices[i] @ tmp => shape (m,) = beta[i,:]
            tmp2 = self.trans_matrices[i].dot(tmp)
            # optional normalization
            norm = tmp2.sum()
            if norm > 0:
                tmp2 /= norm
            self.beta[i, :] = tmp2

    def marginal_of(self, i):
        """
        Return the normalized marginal distribution of X_i:
           p(X_i = k) ~ alpha[i,k]*beta[i,k].
        """
        unnorm = self.alpha[i, :] * self.beta[i, :]
        Z = unnorm.sum()
        if Z > 0:
            return unnorm / Z
        else:
            # if Z=0, either everything is zero or numeric underflow
            # handle carefully; here just return uniform or zeros
            return np.ones(self.m) / self.m

    def assign_value(self, i, value):
        """
        Clamp variable X_i to a given 'value' in [0..m-1].
        Then re-run forward-backward to get updated marginals.
        """
        # Reset unary_factors[i] so that only 'value' is possible
        self.unary_factors[i, :] = 0.0
        self.unary_factors[i, value] = 1.0
        # Re-run
        self.run_forward_backward()

    def sample_sequence(self):
        """
        Sample a configuration X1,...,Xn from left to right:
          1) Sample X1 ~ unary_factors[0] normalized
          2) For i in [0..n-2], sample X_{i+1} ~ trans_matrices[i, X_i, :] * unary_factors[i+1,:]
        Return: a NumPy array of shape (n,) with the sampled states.
        """
        n, m = self.n, self.m
        X = np.zeros(n, dtype=int)

        # 1) Sample X1 from unary_factors[0]
        p1 = self.unary_factors[0, :].copy()
        p1_sum = p1.sum()
        if p1_sum > 0:
            p1 /= p1_sum
        # sample from p1
        X[0] = np.random.choice(m, p=p1)

        # 2) For i in [0..n-2], sample X_{i+1}
        for i in range(n - 1):
            # distribution for X_{i+1} given X_i
            j = X[i]
            # trans_row = trans_matrices[i, j, :] => shape (m,)
            # multiply by unary_factors[i+1, :]
            p_next = self.trans_matrices[i, j, :].copy()
            p_next *= self.unary_factors[i + 1, :]
            p_sum = p_next.sum()
            if p_sum > 0:
                p_next /= p_sum
            X[i + 1] = np.random.choice(m, p=p_next)

        return X


if __name__ == "__main__":
    # Example usage:
    n = 50  # number of variables
    m = 500  # domain size

    # Suppose we have arbitrary unary_factors and transition matrices:
    np.random.seed(42)
    unary_factors = np.random.rand(n, m) + 0.1  # random positive
    trans_matrices = np.random.rand(n - 1, m, m) + 0.1

    bp = ChainBP(unary_factors, trans_matrices)
    t0 = time.perf_counter_ns()
    bp.run_forward_backward()
    t1 = time.perf_counter_ns()
    print(f"Time for forward-backward: {(t1 - t0) / 1_000_000}ms")

    # Print marginals
    # for i in range(n):
    #     marg = bp.marginal_of(i)
    #     print(f"Marginal of X_{i}: {marg}")

    # Clamp X_2 to a specific value
    bp.assign_value(i=2, value=1)
    # Now check marginals again
    for i in range(n):
        marg = bp.marginal_of(i)
        print(f"Marginal of X_{i} after clamping X_2=1: {marg}")

    # Sample a full sequence from left to right

    t0 = time.perf_counter_ns()
    sample = bp.sample_sequence()
    t1 = time.perf_counter_ns()
    print(f"Time to sample: {(t1 - t0) / 1_000_000}ms")
    print("Sampled sequence:", sample)
