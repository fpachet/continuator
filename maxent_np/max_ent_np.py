"""
Compute formula (5) (6) in paper.
https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-08028-4/MediaObjects/41598_2017_8028_MOESM49_ESM.pdf
"""

import numpy as np
import numpy.typing as npt

from core.max_entropy4 import MaxEntropyMelodyGenerator
from maxent_np import NDArrayInt
from maxent_np.preprocess_indices import (
    compute_contexts,
    compute_context_indices,
    compute_partition_context_indices,
)
from utils.profiler import timeit


class MaxEnt:
    PADDING = -1

    # 1D-numpy array of shape (M), training_seq[µ] is the index of the symbol at
    # position µ in the training sequence
    ix_seq: list[int]

    # The number of symbols in the training sequence
    M: int

    # The number of symbols in the vocabulary
    q: int

    # The context width, i.e., the maximum distance from the center of a context
    Kmax: int

    # 3D-numpy array of shape (kmax, q, q), J(k, σ, τ) is the interaction potential
    # between symbols σ and τ at distance k from each other within a context
    J: npt.NDArray[float]

    # 1D-numpy array of shape (q), h[σ] is the bias of symbol σ
    h: np.ndarray

    # 2D-numpy array of shape (M, 2•kmax + 1), contexts[µ] is the context centered
    # around the symbol at position µ
    contexts: NDArrayInt

    # The left-padding index, usually -1
    left_padding: int

    # The right-padding index, usually -2
    right_padding: int

    # The partition function, this is a 1D-array of shape (M)
    Z: npt.NDArray[float]

    Z_ix: tuple[NDArrayInt, ...]
    L_ix1: NDArrayInt
    L_ix: tuple[NDArrayInt, ...]

    # λ in the paper, the lambda regularization parameter
    l: float

    def __init__(
        self,
        index_training_seq: list[int],
        *,
        q: int,
        kmax,
        l=1.0,
    ):
        self.ix_seq: list[int] = index_training_seq
        self.M: int = len(self.ix_seq)
        self.q: int = q
        self.Kmax: int = kmax
        self.l = l

        self.Z = np.zeros(self.M, dtype=float)

        # init h with h[:q] = [0, 1/q, 2/q, ..., 1] and h[q] = PADDING
        # self.h = np.zeros(self.q, dtype=float)
        self.h = np.linspace(0, 1, self.q)

        # init J with j_init and an additional row of zeros at the end and an
        # additional column of zeros at the end of each row

        self.J = np.zeros((self.Kmax, self.q + 1, self.q + 1), dtype=float)
        j_init = np.linspace(0, 1, self.q**2).reshape(self.q, self.q)
        self.J[:, : self.q, : self.q] = j_init

        self.contexts = compute_contexts(
            index_training_seq,
            kmax=self.Kmax,
            padding=MaxEnt.PADDING,
        )

        # get the indices in J of the contexts prepared in such a way that J[self.Z_ix]
        # returns the potential values for all contexts in a single 1D-array
        # see compute_z() for more details on how this is used
        _indices = compute_partition_context_indices(
            self.contexts, q=self.q, kmax=self.Kmax
        )
        _indices = np.swapaxes(_indices, 1, 2)
        _indices = np.reshape(_indices, (self.M, 3, -1))
        _indices = np.swapaxes(_indices, 0, 1)
        _indices = np.reshape(_indices, (3, -1))
        self.Z_ix = tuple(row for row in _indices)

        self.L_ix1 = compute_context_indices(self.contexts, self.Kmax)
        _indices = compute_context_indices(self.contexts, self.Kmax)
        _indices = np.swapaxes(_indices, 0, 1)
        _indices = np.reshape(_indices, (3, -1))
        self.L_ix = tuple(row for row in _indices)

        # get the partition context indices and converts the 4D-array of shape (M, q, 3, 2•kmax) into a list
        # of length M, each element is a list of length q whose elements are tuple of three 2•kmax numpy vectors
        # This is to make J[self.partition_context_indices[mu, sigma]] return the result using numpy matrix indexing
        _partition_context_indices = compute_partition_context_indices(
            self.contexts, q=self.q, kmax=self.Kmax
        )
        self.partition_context_indices = []
        for row_mu in _partition_context_indices:
            matrix_mu_sigma = []
            for col_sigma in row_mu:
                matrix_mu_sigma.append(tuple(col_sigma))
            self.partition_context_indices.append(matrix_mu_sigma)

        self.compute_z()
        print(self.nll())

    @timeit
    def compute_z(self):
        """
        Compute the partition function Z for each context µ in the training sequence.

        Does it without using for loops by using numpy matrix indexing. See how
        self.Z_ix is laid out in the __init__ method for more details.

        Formula (5) in the referenced paper.
        """

        # all_j is essentially J[self.Z_ix] for all mu < self.M and all sigma < self.q
        # first, all_j is a 1D-array of shape (M * q * 2•kmax)
        all_j = self.J[self.Z_ix]
        # then it is reshaped into a 3D-array of shape (M, q, 2•kmax)
        # so all_j[mu, sigma] is the array with all interaction potentials
        all_j = np.reshape(all_j, (self.M, self.q, -1))

        # concatenate the h vector to all the J vectors
        h = np.tile(self.h, (self.M, 1))
        h = h[:, :, None]
        # this is a 3D-array of shape (M, q, 2•kmax + 1)
        h_plus_all_j = np.concatenate((h, all_j), axis=2)

        # formula for Z
        self.Z = np.sum(np.exp(np.sum(h_plus_all_j, axis=2)), axis=1)

    @timeit
    def compute_z_slow(self):
        """DO NOT USE — ONLY HERE FOR TESTING PURPOSES"""
        for mu in range(self.M):
            self.Z[mu] = 0.0
            for sigma in range(self.q):
                indices = self.partition_context_indices[mu][sigma]
                self.Z[mu] += np.exp(self.h[sigma] + self.J[indices].sum())

    @timeit
    def nll(self):
        """
        Compute the negative log-likelihood of the model given the training sequence.

        Formula (6) in the referenced paper.

        Returns:
            a float (the NLL)
        """
        sum_h = self.h[self.ix_seq].sum()
        sum_j = self.J[self.L_ix].sum()
        log_z = np.log(self.Z).sum()
        norm1_j = np.sum(np.abs(self.J))
        return (-(sum_h + sum_j - log_z) + self.l * norm1_j) / self.M

    @timeit
    def grad_loc_field(self):
        """
        Formula (7) in the referenced paper.
        Returns:
            a 1D-numpy array of shape (q)
        """
        dg_dh = np.zeros(self.q)
        for r in range(self.q):
            dg_dh[r] = 0
            for mu in range(self.M):
                if self.ix_seq[mu] == r:
                    dg_dh[r] += 1
                dg_dh[r] -= (1 / self.Z[mu]) * np.exp(
                    self.h[r] + self.J[tuple(self.L_ix1[mu])].sum()
                )
        return -dg_dh / self.M

    @timeit
    def grad_inter_pot(self):
        """
        Formula (8) in the referenced paper.
        Returns:
            a 1D-numpy array of shape (q)
        """
        dg_dJ = np.zeros((self.Kmax, self.q, self.q))
        for k in range(self.Kmax):
            for r in range(self.q):
                for r2 in range(self.q):
                    ...
        return dg_dJ / self.M


if __name__ == "__main__":
    g = MaxEntropyMelodyGenerator("../data/test_sequence_3notes.mid", Kmax=10)

    me = MaxEnt(g.seq, q=g.voc_size, kmax=5)
    print(f"Z = {me.Z}")
    print(f"NLL = {me.nll()}")
    print(f"Loc. grad. = {me.grad_loc_field()}")
