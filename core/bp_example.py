import numpy as np


def forward_pass(transition_matrix, start_state, num_states):
    forward_messages = [{} for _ in range(num_states)]
    forward_messages[0] = {start_state: 1.0}  # Start state is fixed

    for t in range(1, num_states):
        for curr_state in transition_matrix.keys():
            forward_messages[t][curr_state] = sum(
                forward_messages[t - 1].get(prev_state, 0) * transition_matrix[prev_state].get(curr_state, 0)
                for prev_state in transition_matrix.keys()
            )
    return forward_messages


def backward_pass(transition_matrix, end_state, num_states):
    backward_messages = [{} for _ in range(num_states)]
    backward_messages[-1] = {end_state: 1.0}  # End state is fixed

    for t in range(num_states - 2, -1, -1):
        for curr_state in transition_matrix.keys():
            backward_messages[t][curr_state] = sum(
                backward_messages[t + 1].get(next_state, 0) * transition_matrix[curr_state].get(next_state, 0)
                for next_state in transition_matrix.keys()
            )
    return backward_messages


def compute_marginals(forward_messages, backward_messages, num_states, transition_matrix):
    marginals = [{s: 0 for s in transition_matrix.keys()} for _ in range(num_states)]
    for t in range(num_states):
        total = sum(forward_messages[t].get(s, 0) * backward_messages[t].get(s, 0) for s in transition_matrix.keys())
        for s in transition_matrix.keys():
            marginals[t][s] = (forward_messages[t].get(s, 0) * backward_messages[t].get(s, 0)) / total if total > 0 else 0
    return marginals


def sample_sequence(marginals):
    sequence = []
    for t in range(len(marginals)):
        states, probabilities = zip(*marginals[t].items())
        sequence.append(np.random.choice(states, p=probabilities))
    return sequence


# Example with a simple Markov chain
transition_matrix = {
    'A': {'A': 0.7, 'B': 0.2, 'C': 0.1},
    'B': {'A': 0.4, 'B': 0.2, 'C' : 0.4},
    'C': {'A': 0.3, 'B': 0.6, 'C': 0.1}
}
num_states = 10
start_state = 'A'
end_state = 'C'

# Compute messages
forward_messages = forward_pass(transition_matrix, start_state, num_states)
backward_messages = backward_pass(transition_matrix, end_state, num_states)
marginals = compute_marginals(forward_messages, backward_messages, num_states, transition_matrix)

# Sample a sequence
sampled_sequence = sample_sequence(marginals)
print("Generated sequence:", sampled_sequence)
