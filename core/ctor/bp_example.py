import numpy as np


def forward_pass(transition_matrix, start_state, num_states):
    forward_messages = np.zeros((num_states, transition_matrix.shape[0]))
    forward_messages[0, start_state] = 1.0  # Start state is fixed

    for t in range(1, num_states):
        forward_messages[t] = forward_messages[t - 1] @ transition_matrix
    return forward_messages


def backward_pass(transition_matrix, end_state, num_states):
    backward_messages = np.zeros((num_states, transition_matrix.shape[0]))
    backward_messages[-1, end_state] = 1.0  # End state is fixed

    for t in range(num_states - 2, -1, -1):
        backward_messages[t] = transition_matrix @ backward_messages[t + 1]
    return backward_messages


def compute_marginals(forward_messages, backward_messages):
    marginals = forward_messages * backward_messages
    marginals /= marginals.sum(axis=1, keepdims=True)  # Normalize
    return marginals


def sample_sequence(marginals):
    return [np.random.choice(len(marginals[0]), p=marginals[t]) for t in range(len(marginals))]


# Example with a simple Markov chain
transition_matrix = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])
num_states = 5
start_state = 0  # 'A'
end_state = 1  # 'B'

# Compute messages
forward_messages = forward_pass(transition_matrix, start_state, num_states)
backward_messages = backward_pass(transition_matrix, end_state, num_states)
marginals = compute_marginals(forward_messages, backward_messages)

# Sample a sequence
sampled_sequence = sample_sequence(marginals)
print("Generated sequence:", sampled_sequence)
