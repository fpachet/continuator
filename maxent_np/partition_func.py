from typing import Iterable

import numpy as np

from core.max_entropy4 import MaxEntropyMelodyGenerator
from maxent_np.preprocess_indices import compute_contexts, compute_context_indices, \
    compute_partition_context_indices
from utils.profiler import timeit


class MaxEnt:
    # 3D-numpy array of shape (kmax, q, q), J(k, i, j) is the interaction potential between i and j at distance k
    J: np.ndarray[float]
    expJ: np.ndarray[float]
    h: np.ndarray[float]
    training_seq: list[int]
    contexts: np.ndarray[int]
    kmax: int
    left_padding: int
    right_padding: int
    Z: np.ndarray[float]
    M: int

    def __init__(self, index_training_seq: list[int], *, q: int, kmax, left_padding=-1, right_padding=-2):
        self.training_seq: list[int] = index_training_seq
        self.M: int = len(self.training_seq)
        self.q: int = q
        self.kmax: int = kmax

        self.Z = np.zeros(self.M, dtype=float)
        self.h = np.random.random(self.q)
        self.J = np.random.random((self.kmax, self.q, self.q))

        self.left_padding = left_padding
        self.right_padding = right_padding

        self.contexts = compute_contexts(index_training_seq, kmax=self.kmax, left_padding=self.left_padding,
                                         right_padding=self.right_padding)
        self.context_indices = compute_context_indices(self.contexts, self.kmax)

        # get the partition context indices and converts the 4D-array of shape (M, q, 3, 2•kmax) into a list
        # of length M, each element is a list of length q whose elements are tuple of three 2•kmax numpy vectors
        # This is to make J[self.partition_context_indices[mu, sigma]] return the result using numpy matrix indexing
        _partition_context_indices = compute_partition_context_indices(self.contexts, q=self.q, kmax=self.kmax)
        self.partition_context_indices = []
        for row_mu in _partition_context_indices:
            matrix_mu_sigma = []
            for col_sigma in row_mu:
                matrix_mu_sigma.append(tuple(col_sigma))
            self.partition_context_indices.append(matrix_mu_sigma)

        self.compute_Z()

    @timeit
    def compute_Z(self):
        for mu in range(self.M):
            for sigma in range(self.q):
                indices = self.partition_context_indices[mu][sigma]
                # indices = tuple(indices)
                x = self.h[sigma] + self.J[indices].sum()
                self.Z[mu] = np.exp(x)


if __name__ == '__main__':
    g = MaxEntropyMelodyGenerator("../data/bach_partita_mono.midi", Kmax=10)

    print(MaxEnt(g.seq, q=g.voc_size, kmax=10))
