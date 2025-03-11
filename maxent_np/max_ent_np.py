"""
Implementation of paper, using mostly numpy arrays.

https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-08028-4/MediaObjects/41598_2017_8028_MOESM49_ESM.pdf
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from maxent_np import NDArrayInt, NDArrayFloat, FloatType, IdxType
from maxent_np.preprocess_indices import (
    compute_contexts,
    compute_context_indices,
    compute_partition_context_indices,
)


# from utils.profiler import Timeit


class MaxEnt:
    """
    A class to represent a Maximum Entropy model.

    Attributes:
        S: list[int] — 1D-numpy array of shape (M), S[µ] is the index
            of the symbol at position µ in the training sequence.

        M: int — the number of symbols in the training sequence; the typical index used
            to denote an index of an element in the training sequence is µ.

        q: int — the number of symbols in the vocabulary; the typical index used to
            denote an index of a symbol in the vocabulary is σ; if another symbol is
            needed, τ is used.

        K: int — the context width, i.e., the maximum distance from the center of
            a context; the typical index used to denote a distance from the center of a
            context is k.

        J: npt.NDArray[float] — 3D-numpy array of shape (K, q, q), element
            J(k, σ, τ) is the interaction potential between symbols σ and τ at
            distance k from each other within a context.

        h: np.ndarray — 1D-numpy array of shape (q), h[σ] is the bias of symbol σ.

        C: NDArrayInt — 2D-numpy array of shape (M, 2•kmax + 1), contexts[µ] is
            the context centered around the symbol at position µ.

        Z: npt.NDArray[float] — 1D-numpy array of shape (M), Z[µ] is the partition
            function for the context centered around the symbol at position µ.

        context_ix: NDArrayInt — 3D-numpy array of shape (M, 3, 2•kmax), context_ix[µ]
            is the indices of the interaction potentials of the context centered around
            symbol index S[µ].

        partition_ix: NDArrayInt — 4D-numpy array of shape (M, q, 3, 2•kmax),
            partition_ix[µ, σ] is the indices of the interaction potentials of the
            context centered around symbol index S[µ] with the center replaced by
            symbol index σ.

        J5: tuple[NDArrayInt, ...] — a tuple of 1D-numpy arrays of shape (M•q•3•2•kmax),
            J5 is created by reshaping partition_ix into a tuple of 3 1D-arrays; the
            first array contains distances to the center of a context; the second array
            is the index of a left symbol in a context; the third array is the index of
            a right symbol in a context; J5 is used to get in a single numpy access all
            the interaction potentials for all contexts in the training sequence using
            self.J[self.J5]; this is a lot faster than using for loops; this is used to
            compute Formula 5.

        K7: npt.NDArray[bool] — 2D-numpy array of shape (q, M), K7[σ, mu] is True if, and
            only if, S[µ] == σ; this is used to compute Formula 7.
    """

    PADDING = -1

    S: list[int]
    M: int
    q: int
    K: int
    J: npt.NDArray[FloatType]
    h: np.ndarray
    C: NDArrayInt
    Z: npt.NDArray[FloatType]

    context_ix: NDArrayInt
    partition_ix: NDArrayInt
    J5: tuple[NDArrayInt, ...]
    J6: tuple[NDArrayInt, ...]
    J7: tuple[NDArrayInt, ...]

    K7: npt.NDArray[bool]

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
        self.S: NDArrayInt = np.array(index_training_seq, dtype=IdxType)
        self.M: int = len(self.S)
        self.q: int = q
        self.K: int = kmax
        self.l = l

        self.Z = np.zeros(self.M, dtype=FloatType)

        self.h = np.zeros(self.q, dtype=FloatType)

        # init J with j_init and an additional row of zeros at the end and an
        # additional column of zeros at the end of each row
        self.J = np.zeros((self.K, self.q + 1, self.q + 1), dtype=FloatType)

        self.C = compute_contexts(
            index_training_seq,
            kmax=self.K,
            padding=MaxEnt.PADDING,
        )

        self.L_ix_arr = compute_context_indices(self.C, self.K)
        _indices = compute_context_indices(self.C, self.K)
        _indices = np.swapaxes(_indices, 0, 1)
        _indices = np.reshape(_indices, (3, -1))
        self.J5 = tuple(row for row in _indices)

        # get the indices in J of the contexts prepared in such a way that J[self.Z_ix]
        # returns the potential values for all contexts in a single 1D-array
        # see compute_z() for more details on how this is used
        self.partition_ix = compute_partition_context_indices(
            self.C, q=self.q, kmax=self.K
        )
        _indices = self.partition_ix  # (M, q, 3, 2•K)
        _indices = np.swapaxes(_indices, 1, 2)  # (M, 3, q, 2•K)
        _indices = np.reshape(_indices, (self.M, 3, -1))  # (M, 3, 2•K•q)
        _indices = np.swapaxes(_indices, 0, 1)  # (3, M, 2•K•q)
        _indices = np.reshape(_indices, (3, -1))  # (3, M•2•K•q)
        self.J6 = tuple(row for row in _indices)

        _indices = self.partition_ix  # (M, q, 3, 2•K)
        _indices = np.swapaxes(_indices, 0, 1)  # (q, M, 3, 2•K)
        _indices = np.swapaxes(_indices, 1, 2)  # (q, 3, M, 2•K)
        _indices = np.swapaxes(_indices, 0, 1)  # (3, q, M, 2•K)
        _indices = np.reshape(_indices, (3, -1))  # (3, q•M•2•K)
        self.J7 = tuple(row for row in _indices)

        self.K7 = np.arange(self.q)[:, None] == self.S[None, :]

        # get the partition context indices and converts the 4D-array of shape (M, q, 3, 2•kmax) into a list
        # of length M, each element is a list of length q whose elements are tuple of three 2•kmax numpy vectors
        # This is to make J[self.partition_context_indices[mu, sigma]] return the result using numpy matrix indexing
        _partition_context_indices = compute_partition_context_indices(
            self.C, q=self.q, kmax=self.K
        )
        self.partition_context_indices = []
        for row_mu in _partition_context_indices:
            matrix_mu_sigma = []
            for col_sigma in row_mu:
                matrix_mu_sigma.append(tuple(col_sigma))
            self.partition_context_indices.append(matrix_mu_sigma)

        self.checkpoint_index = 0
        self.compute_z()

    def save_checkpoint(self, path: str):
        np.savez(
            path,
            S=self.S,
            M=self.M,
            q=self.q,
            K=self.K,
            J=self.J,
            h=self.h,
        )

    # @Timeit
    def compute_z(self):
        """
        Compute the partition function Z for each context µ in the training sequence.

        Does it without using for loops by using numpy matrix indexing. See how
        self.J6 is laid out in the __init__ method for more details.

        Formula (6) in the referenced paper.
        """

        # all_j is essentially J[self.Z_ix] for all mu < self.M and all sigma < self.q
        # first, all_j is a 1D-array of shape (M * q * 2•kmax)
        all_j = self.J[self.J6]

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

    def nll(self):
        """
        Compute the negative log-likelihood of the model given the training sequence.

        Formula (6) in the referenced paper.

        Returns:
            a float (the NLL)
        """
        sum_h = self.h[self.S].sum()
        sum_j = self.J[self.J5].sum()
        log_z = np.log(self.Z).sum()
        norm1_j = np.sum(np.abs(self.J))

        loss = (-(sum_h + sum_j - log_z) + self.l * norm1_j) / self.M
        print("loss={loss}".format(loss=loss))
        return loss

    # @Timeit
    def _grad_loc_field(self, _j_j7):
        """
        Formula (7) in the referenced paper.

        Returns:
            a 1D-numpy array of shape (q)
        """
        sum_potentials = np.sum(_j_j7.reshape(self.q, self.M, -1), axis=2)
        h_plus_sum_potentials = self.h[:, None] + sum_potentials
        return -(self.K7 - np.exp(h_plus_sum_potentials) / self.Z).sum(axis=1) / self.M

    # @Timeit
    def sum_kronecker_1(self, k):
        row_r = np.hstack(
            [
                np.full((self.q, k + 1), fill_value=False, dtype=bool),
                self.K7[:, : self.M - (k + 1)],
            ]
        )
        row_r2 = self.K7[:]
        return -np.count_nonzero(row_r.reshape(self.q, 1, self.M) * row_r2, axis=2)

    # @Timeit
    def sum_kronecker_2(self, k):
        row_r = self.K7[:]
        row_r2 = np.hstack(
            [
                self.K7[:, k + 1 :],
                np.full((self.q, k + 1), fill_value=False, dtype=bool),
            ]
        )
        return -np.count_nonzero(row_r.reshape(self.q, 1, self.M) * row_r2, axis=2)

    # @Timeit
    def exp1(self, _j_j7):
        kronecker = np.zeros((self.K, self.q, self.M), dtype=bool)
        for k in range(self.K):
            kronecker[k] = np.hstack(
                [
                    np.full((self.q, k + 1), fill_value=False, dtype=bool),
                    self.K7[:, : self.M - (k + 1)],
                ]
            )
        sum_potentials = np.sum(_j_j7.reshape(self.q, self.M, -1), axis=2)
        h_plus_sum_potentials = self.h[:, None] + sum_potentials
        normalized_exp = np.exp(h_plus_sum_potentials) / self.Z
        res = -np.sum(
            kronecker.reshape(self.K, self.q, 1, self.M)
            * normalized_exp.reshape(1, self.q, self.M),
            axis=3,
        )
        return res

    # @Timeit
    def exp2(self, _j_j7):
        kronecker = np.zeros((self.K, self.q, self.M), dtype=bool)
        for k in range(self.K):
            kronecker[k] = np.hstack(
                [
                    self.K7[:, k + 1 :],
                    np.full((self.q, k + 1), fill_value=False, dtype=bool),
                ]
            )
        sum_potentials = np.sum(_j_j7.reshape(self.q, self.M, -1), axis=2)
        h_plus_sum_potentials = self.h[:, None] + sum_potentials
        normalized_exp = np.exp(h_plus_sum_potentials) / self.Z
        res = -np.sum(
            kronecker.reshape(self.K, self.q, 1, self.M)
            * normalized_exp.reshape(1, self.q, self.M),
            axis=3,
        )
        return np.swapaxes(res, 1, 2)

    # @Timeit
    def regularization(self):
        return self.l * np.abs(self.J[:, : self.q, : self.q])

    # @Timeit
    def _grad_inter_pot(self, _j_j7):
        """
        Formula (8) in the referenced paper.
        Returns:
            a 1D-numpy array of shape (q)
        """
        dg_dj = -self.exp1(_j_j7)
        dg_dj -= self.exp2(_j_j7)

        for k in range(self.K):
            dg_dj[k] += self.sum_kronecker_1(k)
            dg_dj[k] += self.sum_kronecker_2(k)

        dg_dj += self.regularization()

        return dg_dj / self.M

    # @Timeit
    def update_arrays_from_params(self, params: NDArrayFloat):
        self.h = params[: self.q]
        self.J[:, : self.q, : self.q] = params[self.q :].reshape(self.K, self.q, self.q)

    # @Timeit
    def arrays_to_params(self):
        return np.concatenate([self.h, self.J[:, : self.q, : self.q].reshape(-1)])

    # @Timeit
    def nll_and_grad(self, params):
        self.update_arrays_from_params(params)
        self.compute_z()
        _j_j7 = self.J[self.J7]
        flat_grad = np.concatenate(
            [self._grad_loc_field(_j_j7), self._grad_inter_pot(_j_j7).reshape(-1)]
        )
        return self.nll(), flat_grad

    def training_callback(self, params):
        self.save_checkpoint(f"./model-checkpoint-{self.checkpoint_index}")
        self.checkpoint_index += 1
        print(self.sample_seq(length=100))

    # @Timeit
    def train(self, max_iter=1000):
        self.checkpoint_index = 0
        params_init = np.zeros(self.q + self.K * self.q * self.q)
        res = minimize(
            self.nll_and_grad,
            params_init,
            method="L-BFGS-B",
            # method="Powell",
            jac=True,
            callback=self.training_callback,
            options={"maxiter": max_iter},
        )
        self.h = res.x[: self.q]
        self.J[:, : self.q, : self.q] = res.x[self.q :].reshape(self.K, self.q, self.q)

    def sum_energy_in_context(
        self, seq: NDArrayInt, ix: int, center: int | None = None
    ):
        energy = 0
        for k in range(self.K):
            energy += self.J[k, seq[ix - (k + 1)], center]
            energy += self.J[k, center, seq[ix + (k + 1)]]
        return energy

    def sample_index_seq(self, length=20, burn_in=1000) -> npt.NDArray:
        # index_seq = np.random.randint(0, self.q, size=length + 2 * self.K)
        index_seq = np.full(length + 2 * self.K, fill_value=0)
        index_seq[: self.K] = MaxEnt.PADDING
        index_seq[-self.K :] = MaxEnt.PADDING

        random_gen = np.random.RandomState(seed=1)
        for _ in range(burn_in):
            pos_in_seq = self.K + random_gen.randint(0, length)
            center = index_seq[pos_in_seq]
            current_energy = self.h[center] + self.sum_energy_in_context(
                index_seq, pos_in_seq, int(center)
            )

            new_center = np.random.randint(0, self.q - 1)
            if new_center >= center:
                new_center += 1

            new_energy = self.h[new_center] + self.sum_energy_in_context(
                index_seq, pos_in_seq, new_center
            )
            acceptance_ratio = min(1, np.exp(new_energy - current_energy))
            if random_gen.random() < acceptance_ratio:
                index_seq[pos_in_seq] = new_center
        return index_seq[self.K : -self.K]
