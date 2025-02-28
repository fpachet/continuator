import numpy as np
from collections import defaultdict
from mido import MidiFile, MidiTrack, Message


def extract_pitch_sequence(midi_path):
    mid = MidiFile(midi_path)
    pitches = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pitches.append(msg.note)
    return pitches


def estimate_transition_matrix(pitch_sequence):
    transitions = defaultdict(lambda: defaultdict(int))
    states = set(pitch_sequence)

    for i in range(len(pitch_sequence) - 1):
        transitions[pitch_sequence[i]][pitch_sequence[i + 1]] += 1

    states = sorted(states)
    state_index = {s: i for i, s in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))

    for s1 in transitions:
        total = sum(transitions[s1].values())
        for s2 in transitions[s1]:
            matrix[state_index[s1], state_index[s2]] = transitions[s1][s2] / total

    return matrix, state_index


def sample_sequence(transition_matrix, start_state, end_state, num_states):
    sequence = [start_state]

    for t in range(1, num_states - 1):
        sub_matrix = transition_matrix.copy()
        # sub_matrix[:, sequence] = 0  # Remove already chosen states
        # sub_matrix /= sub_matrix.sum(axis=1, keepdims=True)  # Normalize

        forward_messages = np.zeros((num_states - t, transition_matrix.shape[0]))
        forward_messages[0, sequence[0]] = 1.0  # Start state is fixed

        for i in range(1, num_states - t):
            forward_messages[i] = forward_messages[i - 1] @ sub_matrix

        backward_messages = np.zeros((num_states - t, transition_matrix.shape[0]))
        backward_messages[-1, end_state] = 1.0  # End state is fixed

        for i in range(num_states - t - 2, -1, -1):
            backward_messages[i] = sub_matrix @ backward_messages[i + 1]

        marginals = forward_messages * backward_messages
        marginals /= marginals.sum(axis=1, keepdims=True)  # Normalize

        states = np.arange(transition_matrix.shape[0])
        next_state = np.random.choice(states, p=marginals[0])
        sequence.append(next_state)

    sequence.append(end_state)
    return sequence


def save_midi(sequence, state_index, output_path="generated.mid"):
    reverse_index = {v: k for k, v in state_index.items()}
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for pitch in sequence:
        note = reverse_index[pitch]
        track.append(Message("note_on", note=note, velocity=64, time=200))
        track.append(Message("note_off", note=note, velocity=64, time=200))

    mid.save(output_path)


# Example usage
midi_path = "../../data/ascending_chromatic_c.mid"  # Replace with actual MIDI file path
pitch_sequence = extract_pitch_sequence(midi_path)
transition_matrix, state_index = estimate_transition_matrix(pitch_sequence)
num_states = 10
start_state = state_index[pitch_sequence[0]]
end_state = state_index[pitch_sequence[-1]]

sampled_sequence = sample_sequence(
    transition_matrix, start_state, end_state, num_states
)
print(
    "Generated pitch sequence:", [list(state_index.keys())[s] for s in sampled_sequence]
)

# Save to MIDI file
save_midi(sampled_sequence, state_index, "../../data/generated.mid")
