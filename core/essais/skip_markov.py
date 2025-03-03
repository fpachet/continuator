import mido
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
import random

def train_skip_markov(train_seq, skip):
    """
    Train a first-order skip=Δ Markov model on the given sequence.

    Args:
        train_seq: list of events/notes, e.g. [60, 62, 64, 62, ...]
        skip: integer Δ indicating the skip distance

    Returns:
        A dict of dicts: model[prev_note][next_note] = probability
    """
    transition_counts = defaultdict(lambda: defaultdict(float))

    # Count transitions: note_t -> note_{t+skip}
    for t in range(len(train_seq) - skip):
        prev_note = train_seq[t]
        next_note = train_seq[t + skip]
        transition_counts[prev_note][next_note] += 1.0

    # Convert counts to probabilities
    model = {}
    for prev_note, next_counts in transition_counts.items():
        total = sum(next_counts.values())
        model[prev_note] = {}
        for nxt, cnt in next_counts.items():
            model[prev_note][nxt] = cnt / total

    return model


def train_all_skip_models(train_seq, skip_distances):
    """
    Train multiple skip=Δ models, one for each Δ in skip_distances.

    Returns:
        A dict skip_models where skip_models[d] is the Markov model
        trained with skip=d.
    """
    skip_models = {}
    for d in skip_distances:
        skip_models[d] = train_skip_markov(train_seq, d)
    return skip_models



def build_mixture_matrix(train_seq, skip_models, skip_distances):
    """
    Build a matrix P of shape (T-1, D) for mixture fitting.

    P[t, j] = the probability that skip-model j assigns
               to the *actual* next note x_{t+1},
               given x_{t+1 - skip_d} (if valid).

    If (t+1 - skip_d) < 0, we set P[t, j] = 0.
    """
    T = len(train_seq)
    D = len(skip_distances)
    P = np.zeros((T - 1, D))

    # We'll also build a quick index for skip_distances
    # so skip_distances[j] is the j-th skip.
    for j, d in enumerate(skip_distances):
        model = skip_models[d]
        for t in range(T - 1):
            context_idx = t + 1 - d
            if context_idx >= 0:
                prev_note = train_seq[context_idx]
                next_note = train_seq[t + 1]
                # Probability from the skip model
                # If we don't have prev_note in model, prob=0
                if prev_note in model and next_note in model[prev_note]:
                    P[t, j] = model[prev_note][next_note]
                else:
                    P[t, j] = 0.0
            else:
                # Invalid context
                P[t, j] = 0.0
    return P


def mixture_objective_and_grad(u, P):
    """
    For mixture of D skip models, each row t in P has P[t, j] = p_j(t).

    u: 1D array of shape (D,) of logits (unconstrained).
    P: 2D array of shape (T-1, D).

    Returns:
        (neg_log_likelihood, grad_u)
        so we can minimize the negative log-likelihood.
    """
    D = len(u)
    T_minus_1 = P.shape[0]

    # Softmax to get alpha
    exp_u = np.exp(u)
    sum_exp = np.sum(exp_u)
    alpha = exp_u / sum_exp  # shape (D,)

    # For each time t, mixture sum: s(t) = sum_j alpha_j * p_j(t)
    s = P @ alpha  # shape (T_minus_1,)

    # Avoid log(0) by adding a tiny epsilon if needed
    eps = 1e-12
    s = np.clip(s, eps, None)

    # Log-likelihood
    L = np.sum(np.log(s))  # sum_t ln( s(t) )
    neg_L = -L  # we want to minimize negative LL

    # Gradient
    grad = np.zeros(D)
    for m in range(D):
        alpha_m = alpha[m]
        # partial derivative is sum_t of alpha_m * (P[t,m] - s[t]) / s[t]
        # Then we must negate it for negative log-likelihood
        grad_part = alpha_m * (P[:, m] - s) / s
        grad[m] = np.sum(grad_part)

    grad = -grad
    return neg_L, grad


def fit_mixture_weights(P, method='BFGS'):
    """
    Fit mixture weights alpha via gradient-based optimization.

    Returns:
        alpha (1D numpy array of shape (D,)) summing to 1.
    """
    D = P.shape[1]
    # Initialize logits u randomly (e.g. zero or small noise)
    u_init = np.zeros(D)

    result = minimize(
        fun=mixture_objective_and_grad,
        x0=u_init,
        args=(P,),
        jac=True,  # we provide the gradient
        method=method
    )

    # Convert final u -> alpha via softmax
    u_opt = result.x
    exp_u = np.exp(u_opt)
    alpha = exp_u / np.sum(exp_u)
    return alpha


def generate_sequence(skip_models, skip_distances, alpha, seed_note, gen_len=20):
    """
    Generate a new sequence of length gen_len.

    Args:
        skip_models: dict of {d: model}, each model is [prev_note]->{next_note->prob}
        skip_distances: list of skip distances [d1, d2, ...]
        alpha: array of mixture weights shape (D,)
        seed_note: the initial note to start generation
        gen_len: length of generated sequence

    Returns:
        A list of notes (sequence).
    """
    generated = [seed_note]
    D = len(alpha)

    # Collect all possible notes from training for fallback
    # (in case we get no valid distribution from the models).
    # We'll union all notes in all skip models:
    all_notes = set()
    for m in skip_models.values():
        all_notes.update(m.keys())
        for d in m.values():
            all_notes.update(d.keys())
    all_notes = sorted(all_notes)
    vocab_size = len(all_notes)

    for t in range(gen_len - 1):
        # We'll build a mixture distribution over next notes
        mixture_dist = defaultdict(float)

        # For each skip model j
        for j, d in enumerate(skip_distances):
            idx = (len(generated) ) - d
            if idx >= 0:
                prev_note = generated[idx]
                # If the model has prev_note, retrieve its distribution
                if prev_note in skip_models[d]:
                    local_dist = skip_models[d][prev_note]
                    # Merge into mixture_dist
                    for candidate_next, p_local in local_dist.items():
                        mixture_dist[candidate_next] += alpha[j] * p_local
            else:
                # skip this model if no context
                pass

        # Now normalize mixture_dist (in case not summing to 1)
        total_prob = sum(mixture_dist.values())
        if total_prob < 1e-12:
            print('small proba')
            # fallback: uniform over all notes
            for n in all_notes:
                mixture_dist[n] = 1.0 / vocab_size
            total_prob = 1.0

        # Create a cumulative distribution for sampling
        for k in mixture_dist:
            mixture_dist[k] /= total_prob

        # Sample from mixture_dist
        notes_list = list(mixture_dist.keys())
        probs_list = np.array([mixture_dist[n] for n in notes_list])
        cumsum_probs = np.cumsum(probs_list)
        r = random.random()
        idx_sample = np.searchsorted(cumsum_probs, r)
        chosen_note = notes_list[idx_sample]

        generated.append(chosen_note)

    return generated

def extract_notes(path) -> list[int]:
    """Extracts MIDI note sequence from a MIDI file."""
    mid = mido.MidiFile(path)
    notes = []
    if (len(mid.tracks)) == 1:
        track = mid.tracks[0]
    else:
        track = mid.tracks[1]
    for msg in track:
        if msg.type == "note_on" and msg.velocity > 0:
            notes.append(msg.note)
    return notes

def save_midi(sequence, output_file="../../data/skip_markov_generated.mid"):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for note in sequence:
        track.append(mido.Message("note_on", note=note, velocity=64, time=0))
        track.append(mido.Message("note_off", note=note, velocity=64, time=120))
    mid.save(output_file)

def main_example():
    # path = "../../data/bach_partita_mono.midi"
    path = "../../data/prelude_c.mid"
    train_seq = extract_notes(path)

    # with open('../../data/proust_du_cote.txt', 'r') as file:
    #     recherche = file.read().rstrip()
    # train_seq = list(recherche)

    # train_seq = [60, 62, 64, 62, 60, 60, 67, 65, 64, 62, 60]
    # train_seq = [1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 5, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 5, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 5, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 5, 1, 2, 4, 1, 2, 3]
    # 2) Choose skip distances to try
    # skip_distances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    skip_distances = np.arange(1, 4, 1)
    # skip_distances = [1]

    # Train the skip models
    skip_models = train_all_skip_models(train_seq, skip_distances)

    # Build P matrix
    P = build_mixture_matrix(train_seq, skip_models, skip_distances)

    # Fit mixture weights alpha
    alpha = fit_mixture_weights(P, method='BFGS')
    print("Learned mixture weights:", alpha)

    # Generate new sequence
    seed_note = 60
    gen_seq = generate_sequence(skip_models, skip_distances, alpha, seed_note, gen_len=200)
    # print (gen_seq)

    # result = ''.join(gen_seq)
    # print(result)
    save_midi(gen_seq)

# Uncomment to run the example:
if __name__ == "__main__":
    main_example()
