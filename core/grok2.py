import numpy as np


def sample_segment(P, a, b, segment_length):
    """
    Sample internal variables X_2, ..., X_{segment_length-1} given X_1 = a and X_{segment_length} = b.

    Parameters:
    - P: K x K transition matrix
    - a: state at the start of the segment (X_1)
    - b: state at the end of the segment (X_{segment_length})
    - segment_length: length of the segment (≥ 2)

    Returns:
    - List of sampled states [X_2, ..., X_{segment_length-1}]
    """
    if segment_length < 2:
        raise ValueError("Segment length must be at least 2")
    if segment_length == 2:
        return []  # No internal variables to sample

    K = P.shape[0]
    # Forward messages: m[t-2] is m_{1→t} for t=2 to segment_length-1
    m = P[a, :]  # m_{1→2} = P(X_2 | X_1 = a)
    m_list = [m]
    for t in range(3, segment_length):
        m = m @ P  # m_{1→t} = sum_{x_{t-1}} P(x_t | x_{t-1}) m_{1→t-1}
        m_list.append(m)

    # Backward sampling
    sequence = []
    s_next = b  # Start with X_{segment_length} = b
    for t in range(segment_length - 1, 1, -1):  # t = segment_length-1 down to 2
        # P(X_t | X_1 = a, X_{t+1} = s_next) ∝ m_{1→t} * P(X_{t+1} | X_t)
        dist = m_list[t - 2] * P[:, s_next]
        dist = dist / np.sum(dist)  # Normalize
        s_t = np.random.choice(K, p=dist)
        sequence.insert(0, s_t)
        s_next = s_t
    return sequence


def sample_sequence_with_constraints(P, constraints, n):
    """
    Sample a sequence of length n with constraints at arbitrary positions.

    Parameters:
    - P: K x K transition matrix
    - constraints: dict {position: value} with 1-based positions (1 to n)
    - n: total sequence length

    Returns:
    - List of length n with constrained positions set and others sampled
    """
    K = P.shape[0]
    sequence = [None] * n

    # Set constrained positions (convert 1-based to 0-based indices)
    for pos, val in constraints.items():
        if not (1 <= pos <= n):
            raise ValueError(f"Position {pos} out of range [1, {n}]")
        if not (0 <= val < K):
            raise ValueError(f"State {val} out of range [0, {K - 1}]")
        sequence[pos - 1] = val

    # Sorted constrained positions (1-based)
    c_positions = sorted(constraints.keys()) if constraints else []

    # Handle segment before first constraint (if position 1 is not constrained)
    if c_positions and c_positions[0] > 1:
        # Sample from 1 to c_positions[0] - 1, conditioned only on X_{c_positions[0]}
        start = 1
        end = c_positions[0]
        b = constraints[end]
        # Assume uniform initial distribution at X_1 since no constraint
        m = np.ones(K) / K  # Initial distribution
        m_list = [m]
        for t in range(start + 1, end):  # Forward messages up to end-1
            m = m @ P
            m_list.append(m)
        segment = []
        s_next = b
        for t in range(end - 1, start, -1):
            dist = m_list[t - start] * P[:, s_next]
            dist = dist / np.sum(dist)
            s_t = np.random.choice(K, p=dist)
            segment.insert(0, s_t)
            s_next = s_t
        # Sample X_1
        dist = np.ones(K) / K * P[:, s_next]
        dist = dist / np.sum(dist)
        s_t = np.random.choice(K, p=dist)
        segment.insert(0, s_t)
        sequence[0 : end - 1] = segment

    # Sample segments between consecutive constraints
    for k in range(len(c_positions) - 1):
        i = c_positions[k]  # Start position (1-based)
        j = c_positions[k + 1]  # End position (1-based)
        a = constraints[i]
        b = constraints[j]
        segment_length = j - i + 1
        if segment_length > 2:
            sampled_segment = sample_segment(P, a, b, segment_length)
            sequence[i : j - 1] = sampled_segment

    # Handle segment after last constraint (if position n is not constrained)
    if c_positions and c_positions[-1] < n:
        # Sample from c_positions[-1] + 1 to n, conditioned only on X_{c_positions[-1]}
        start = c_positions[-1]
        end = n
        a = constraints[start]
        segment_length = end - start + 1
        if segment_length > 1:
            # Forward messages from start to end-1
            m = P[a, :]
            m_list = [m]
            for t in range(start + 2, end + 1):
                m = m @ P
                m_list.append(m)
            segment = []
            # Sample X_n first (no constraint), assume reverse transition or uniform
            # For simplicity, use last forward message as approximation
            if segment_length == 2:
                dist = P[a, :]
            else:
                dist = m_list[-1]
            dist = dist / np.sum(dist)
            s_next = np.random.choice(K, p=dist)
            segment.insert(0, s_next)
            # Backward sample the rest
            for t in range(end - 1, start + 1, -1):
                dist = m_list[t - start - 1] * P[:, s_next]
                dist = dist / np.sum(dist)
                s_t = np.random.choice(K, p=dist)
                segment.insert(0, s_t)
            sequence[start:n] = segment

    # If no constraints at all, sample the entire sequence
    if not c_positions:
        m = np.ones(K) / K
        m_list = [m]
        for t in range(2, n + 1):
            m = m @ P
            m_list.append(m)
        sequence = []
        s_next = np.random.choice(K, p=m_list[-1] / np.sum(m_list[-1]))
        sequence.insert(0, s_next)
        for t in range(n - 1, 1, -1):
            dist = m_list[t - 1] * P[:, s_next]
            dist = dist / np.sum(dist)
            s_t = np.random.choice(K, p=dist)
            sequence.insert(0, s_t)
        dist = np.ones(K) / K * P[:, s_next]
        dist = dist / np.sum(dist)
        s_t = np.random.choice(K, p=dist)
        sequence.insert(0, s_t)
        return sequence

    return sequence


# Example usage
if __name__ == "__main__":
    # Example transition matrix (3 states)
    # P = np.array(
    #     [
    #         [0, 1, 0, 0, 0, 0],
    #         [0, 0.3, 0.4, 0, 0.3, 0],
    #         [0.1, 0.5, 0.4, 0, 0, 0],
    #         [0.3, 0.4, 0, 0, 0, 0.3],
    #         [0.1, 0.4, 0, 0.5, 0, 0],
    #         [0, 0.3, 0, 0.5, 0.2, 0],
    #     ]
    # )
    P = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
        ]
    )
    n = 4

    # Test with constraints at various positions
    constraints = {}  # X_1 = 0, X_3 = 1, others free
    seq = sample_sequence_with_constraints(P, constraints, n)
    print(f"Sequence with constraints {constraints}:", seq)
