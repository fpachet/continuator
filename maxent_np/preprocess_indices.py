import numpy as np

from maxent_np import NDArrayInt
from utils.profiler import timeit


def compute_contexts(idx_seq: list[int], /, *, kmax: int, padding=-1) -> NDArrayInt:
    """
    Compute all contexts for a given index sequence.

    The input sequence contains indices of elements of the alphabet.
    Each context is a 2•kmax + 1 window around a sequence element.
    Note that the input sequence is left- and right-padded with kmax left_padding and
    right_padding indices respectively, to ensure that the first and last elements
    lead to contexts of length 2•kmax + 1.

    Args:
        idx_seq: the index sequence
        kmax: defines the size of contexts, i.e., 2•kmax + 1
        padding: the index used for padding

    Returns:
        a 2D-numpy array of shape (m, 2•kmax + 1) where m is the length of the
        input sequence

    """
    m = len(idx_seq)  # length of the input sequence
    l = 2 * kmax + 1  # length of the context
    c = np.zeros((m, l), dtype=int)  # the context matrix

    # left-pad and right-pad the input sequence
    padded_seq = np.concatenate(
        [np.array([padding] * kmax), idx_seq, np.array([padding] * kmax)]
    )

    for i in np.arange(m):
        c[i, :] = padded_seq[i : i + l]

    return c


def compute_context_indices(contexts: NDArrayInt, kmax: int = 0) -> NDArrayInt:
    """
    Compute the indices of each context in J, the 3D-array of interaction potentials.

    The result R is a 3D-array of shape (m, 3, 2•kmax). For a given context µ,
    R[µ] is a (3, 2•kmax) array such that J[R[µ]] is the interaction potential values
    corresponding to the context µ. If the context is [A, B, C, D, E, F, G] (assuming
    kmax = 3), then D is s_0 (the center of the context) and
        R[µ, 0] = [0, 0, 1, 1, 2, 2]
        R[µ, 1] = [C, D, B, D, A, D]
        R[µ, 2] = [D, E, D, F, D, G]

    Since J is indexed by K, left index, right index, J[R[µ]] will give the interaction
    potentials:
        J[0, C, D], J[0, D, E] // the potentials at distance 1 from the center
        J[1, B, D], J[1, D, F] // the potentials at distance 2 from the center
        J[2, A, D], J[2, D, G] // the potentials at distance 3 from the center

    Therefore J[R[µ]].sum() will give the total interaction potential for the context
    µ, as computed in the Sum-Energy in Formula 5 in the paper
    https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-08028-4/MediaObjects/41598_2017_8028_MOESM49_ESM.pdf

    Args:
        contexts: the 2D-array of contexts
        kmax: the maximum distance from the center of the context

    Returns:
        a 3D-array of shape (m, 3, 2•kmax)
    """
    m, ctx_length = contexts.shape
    kmax = kmax or (ctx_length - 1) // 2

    k_indices = np.tile(np.arange(kmax, dtype=int).repeat(2), m).reshape(m, -1)

    left_indices = np.zeros(2 * kmax, dtype=int)
    left_indices[0::2] = np.arange(kmax - 1, -1, -1)
    left_indices[1::2] = kmax

    right_indices = np.zeros(2 * kmax, dtype=int)
    right_indices[1::2] = np.arange(kmax + 1, 2 * kmax + 1)
    right_indices[0::2] = kmax

    left_indices = contexts[:, left_indices]
    right_indices = contexts[:, right_indices]

    indices = np.zeros((m, 3, 2 * kmax), dtype=int)

    indices[:, 0, :] = k_indices
    indices[:, 1, :] = left_indices
    indices[:, 2, :] = right_indices

    return indices


def compute_context_indices_naive(contexts, kmax=0):
    """
    same as compute_context_indices2 except much slower

    kept as a reference since it is easier to understand
    """
    m, l = contexts.shape
    kmax = kmax or (l - 1) // 2
    result = np.zeros((m, 3, 2 * kmax), dtype=int)
    for i in range(m):
        for k in range(kmax):
            s_0 = contexts[i, kmax]
            s_k = contexts[i, kmax + k + 1]
            s_mk = contexts[i, kmax - k - 1]
            result[i, 0, 2 * k : 2 * (k + 1)] = k
            result[i, 1, 2 * k] = s_mk
            result[i, 1, 2 * k + 1] = s_0
            result[i, 2, 2 * k] = s_0
            result[i, 2, 2 * k + 1] = s_k
    return result


def compute_partition_context_indices(
    contexts, /, *, q: int, kmax: int = 0
) -> NDArrayInt:
    """
    Compute the indices of each context in J with respect to each 0 ≤ sigma < q.

    The result R is a 4D-array of shape (m, q, 3, 2•kmax). For a given context µ, and
    an element index σ (0 ≤ σ < q), R[µ, σ] is a (3, 2•kmax) array such that J[R[µ, σ]]
    is the interaction potential values corresponding to the context µ if the
    context center is σ. If the context is [A, B, C, D, E, F, G] (assuming
    kmax = 3), then for a given σ:
        R[µ, σ, 0] = [0, 0, 1, 1, 2, 2]
        R[µ, σ, 1] = [C, σ, B, σ, A, σ]
        R[µ, σ, 2] = [σ, E, σ, F, σ, G]

    Since J is indexed by K, left index, right index, J[R[µ]] will give the interaction
    potentials:
        J[0, C, σ], J[0, σ, E]
        J[1, B, σ], J[1, σ, F]
        J[2, A, σ], J[2, σ, G]

    Therefore J[R[µ, σ]].sum() will give the total interaction potential for the context
    µ and elements index σ, as computer in the Partition-Function, Formula (6) in the paper
    https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-08028-4/MediaObjects/41598_2017_8028_MOESM49_ESM.pdf.

    Args:
        contexts: the 2D-array of contexts
        q: the size of the alphabet
        kmax: the maximum distance from the center of the context

    Returns:
        a 4D-array fo shape (m, q, 3, 2•kmax)
    """
    indices = compute_context_indices(contexts, kmax)
    normalization_indices = np.tile(
        indices.reshape(indices.shape[0], 1, *indices.shape[1:]), (1, q, 1, 1)
    )
    for sigma in range(q):
        normalization_indices[:, sigma, 1, 1::2] = sigma
        normalization_indices[:, sigma, 2, 0::2] = sigma

    return normalization_indices


if __name__ == "__main__":
    mat = np.arange(28).reshape((7, 4))
    g = np.random.default_rng(10)
    q = 3
    kmax = 3
    seq = g.integers(0, q, 10)
    ctx = compute_contexts(seq, kmax=kmax)
    j_idx = compute_context_indices(ctx, kmax=kmax)
    z_idx = compute_partition_context_indices(ctx, q=q, kmax=kmax)
