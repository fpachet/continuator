"""
Methods that were implemented in MaxEnt (NumPy version) and that were replaced by
faster implementations using vectorized methods.

They are kept here as they are much easier to understand.
"""

from utils.profiler import Timeit
import numpy as np

"""
Methods that could be faster, but are not in practice
"""


@Timeit
def sum_kronecker_1_experimental(self):
    kronecker = np.zeros((self.K, self.q, self.M), dtype=bool)
    for k in range(self.K):
        kronecker[k] = np.hstack(
            [
                np.full((self.q, k + 1), fill_value=False, dtype=bool),
                self.K7[:, : self.M - (k + 1)],
            ]
        )
    row_r2 = self.K7[:]
    return -np.count_nonzero(
        kronecker.reshape(self.K, self.q, 1, self.M)
        * row_r2.reshape(1, self.q, self.M),
        axis=3,
    )


@Timeit
def sum_kronecker_2_experimental(self):
    row_r = self.K7[:]
    kronecker = np.zeros((self.K, self.q, self.M), dtype=bool)
    for k in range(self.K):
        kronecker[k] = np.hstack(
            [
                self.K7[:, k + 1 :],
                np.full((self.q, k + 1), fill_value=False, dtype=bool),
            ]
        )
    return -np.count_nonzero(
        row_r.reshape(self.q, 1, self.M) * kronecker.reshape(self.K, 1, self.q, self.M),
        axis=3,
    )


@Timeit
def compute_z_slow(self):
    """DO NOT USE — ONLY HERE FOR TESTING PURPOSES"""
    for mu in range(self.M):
        self.Z[mu] = 0.0
        for sigma in range(self.q):
            indices = self.partition_context_indices[mu][sigma]
            self.Z[mu] += np.exp(self.h[sigma] + self.J[indices].sum())


@Timeit
def regularization_slow(self, k, r, r2):
    return self.l * np.abs(self.J[k, r, r2])


@Timeit
def grad_inter_pot_slow(self):
    """
    Formula (8) in the referenced paper.
    Returns:
        a 1D-numpy array of shape (q)
    """
    dg_dJ = np.zeros((self.K, self.q, self.q))
    for k in range(self.K):
        for r in range(self.q):
            for r2 in range(self.q):
                dg_dJ[k, r, r2] += (
                    self.sum_kronecker_1_slow(k, r, r2)
                    + self.sum_kronecker_2_slow(k, r, r2)
                    - self.exp1_slow(k, r, r2)
                    - self.exp2_slow(k, r, r2)
                    + self.regularization_slow(k, r, r2)
                )
    return -dg_dJ / self.M


@Timeit
def exp2_slow(self, k, r, r2):
    s = 0
    for mu in range(self.M):
        if self.C[mu, self.K + (k + 1)] == r2:
            s += (
                np.exp(self.h[r] + self.J[tuple(self.partition_ix[mu, r])].sum())
                / self.Z[mu]
            )
    return -s


@Timeit
def exp2(self, k, r, r2):
    kronecker = np.concatenate(
        [
            self.K7[r2, k + 1 :],
            np.full(k + 1, fill_value=False, dtype=bool),
        ]
    )
    sum_potentials = np.sum(self.J[self.J7].reshape(self.q, self.M, -1), axis=2)
    h_plus_sum_potentials = self.h[:, None] + sum_potentials
    normalized_exp = np.exp(h_plus_sum_potentials) / self.Z
    return -np.sum(kronecker * normalized_exp[r])


@Timeit
def exp1_slow(self, k, r, r2):
    s = 0
    for mu in range(self.M):
        if self.C[mu, self.K - (k + 1)] == r:
            s += (
                np.exp(self.h[r2] + self.J[tuple(self.partition_ix[mu, r2])].sum())
                / self.Z[mu]
            )
    return -s


@Timeit
def exp1(self, k, r):
    kronecker = np.concatenate(
        [
            np.full(k + 1, fill_value=False, dtype=bool),
            self.K7[r, : self.M - (k + 1)],
        ]
    )
    sum_potentials = np.sum(self.J[self.J7].reshape(self.q, self.M, -1), axis=2)
    h_plus_sum_potentials = self.h[:, None] + sum_potentials
    normalized_exp = np.exp(h_plus_sum_potentials) / self.Z
    return -np.sum(kronecker * normalized_exp, axis=1)


@Timeit
def sum_kronecker_2_slow(self, k, r, r2):
    s = 0
    for mu in range(self.M):
        s += (self.C[mu, self.K] == r) * (self.C[mu, self.K + (k + 1)] == r2)
    return -s


@Timeit
def sum_kronecker_2(self, k, r, r2):
    row_r = self.K7[r]
    row_r2 = np.concatenate(
        [
            self.K7[r2, k + 1 :],
            np.full(k + 1, fill_value=False, dtype=bool),
        ]
    )
    return -np.count_nonzero(row_r * row_r2)


@Timeit
def sum_kronecker_1(self, k, r, r2):
    row_r = np.concatenate(
        [
            np.full(k + 1, fill_value=False, dtype=bool),
            self.K7[r, : self.M - (k + 1)],
        ]
    )
    row_r2 = self.K7[r2]
    return -np.count_nonzero(row_r * row_r2)


@Timeit
def sum_kronecker_1_slow(self, k, r, r2):
    s = 0
    for mu in range(self.M):
        s += (self.C[mu, self.K - (k + 1)] == r) * (self.C[mu, self.K] == r2)
    return -s


@Timeit
def grad_loc_field_slow(self):
    """
    DO NOT USE — ONLY HERE FOR TESTING PURPOSES
    """
    dg_dh = np.zeros(self.q)
    for r in range(self.q):
        dg_dh[r] = 0
        for mu in range(self.M):
            if self.S[mu] == r:
                dg_dh[r] += 1
            dg_dh[r] -= (1 / self.Z[mu]) * np.exp(
                self.h[r] + self.J[tuple(self.partition_ix[mu, r])].sum()
            )
    return -dg_dh / self.M
