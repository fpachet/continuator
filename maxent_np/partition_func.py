"""
Compute formula (5) (6) in paper.
https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-08028-4/MediaObjects/41598_2017_8028_MOESM49_ESM.pdf
"""

import numpy as np
import numpy.typing as npt

from core.max_entropy4 import MaxEntropyMelodyGenerator
from maxent_np.preprocess_indices import (
    compute_contexts,
    compute_context_indices,
    compute_partition_context_indices,
)
from utils.profiler import timeit


class MaxEnt:
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
    contexts: np.ndarray[int]

    # The left-padding index, usually -1
    left_padding: int

    # The right-padding index, usually -2
    right_padding: int

    # The partition function, this is a 1D-array of shape (M)
    Z: npt.NDArray[float]

    Z_ix: tuple[npt.NDArray[int], ...]

    L_ix1: tuple[npt.NDArray[int], ...]
    L_ix: tuple[npt.NDArray[int], ...]

    # λ in the paper, the lambda regularization parameter
    l: float

    def __init__(
        self,
        index_training_seq: list[int],
        *,
        q: int,
        kmax,
        left_padding=-1,
        right_padding=-2,
        l=1.0,
    ):
        self.ix_seq: list[int] = index_training_seq
        self.M: int = len(self.ix_seq)
        self.q: int = q
        self.Kmax: int = kmax
        self.l = l

        self.Z = np.zeros(self.M, dtype=float)
        # self.h = np.random.random(self.q)
        # self.J = np.random.random((self.kmax, self.q, self.q))
        self.h = np.linspace(0, 1, self.q)
        self.J = np.zeros((self.Kmax, self.q * self.q))
        self.J[:] = np.linspace(0, 1, self.q**2)
        self.J = np.reshape(self.J, (self.Kmax, self.q, self.q))

        self.left_padding = left_padding
        self.right_padding = right_padding

        self.contexts = compute_contexts(
            index_training_seq,
            kmax=self.Kmax,
            left_padding=self.left_padding,
            right_padding=self.right_padding,
        )
        self.context_indices = compute_context_indices(self.contexts, self.Kmax)

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
        self.Z_ix = tuple(_indices)

        self.L_ix1 = compute_context_indices(self.contexts, self.Kmax)
        _indices = compute_context_indices(self.contexts, self.Kmax)
        _indices = np.swapaxes(_indices, 0, 1)
        _indices = np.reshape(_indices, (3, -1))
        self.L_ix = tuple(_indices)

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
        for mu in range(self.M):
            self.Z[mu] = 0.0
            for sigma in range(self.q):
                indices = self.partition_context_indices[mu][sigma]
                self.Z[mu] += np.exp(self.h[sigma] + self.J[indices].sum())

    @timeit
    def nll(self):
        sum_h = self.h[self.ix_seq].sum()
        sum_j = self.J[self.L_ix].sum()
        log_z = np.log(self.Z).sum()
        norm1_j = np.sum(np.abs(self.J))
        return (-(sum_h + sum_j - log_z) + self.l * norm1_j) / self.M


if __name__ == "__main__":
    g = MaxEntropyMelodyGenerator("../data/bach_partita_mono.midi", Kmax=10)

    me = MaxEnt(g.seq, q=g.voc_size, kmax=10)
    # print(me.Z)
