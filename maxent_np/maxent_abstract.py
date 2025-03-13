import random
from typing import Iterable


class MaxEnt:
    Kmax: int

    def training_index_sequence(self) -> list[int]:
        pass

    def train(self, max_item=1000):
        pass

    def sum_energy_in_context(self, J, context, center):
        """
        Compute the energy of the context. The

        Args:
            J:
            context:
            center:

        Returns:

        """
        energy = 0
        for k in range(self.Kmax):
            j_k = J[k]
            if -k - 1 in context:
                energy += j_k[context.get(-k - 1), center]
            if k + 1 in context:
                energy += j_k[center, context.get(k + 1)]
        return energy

    def sample_seq(self, length=20, burn_in=1000):

        index_sequence = [
            random.choice(self.training_index_sequence()) for _ in range(length)
        ]
        for _ in range(burn_in):
            idx = random.randint(0, length - 1)
            current_note = index_sequence[idx]
            context = self.build_context(index_sequence, idx)
            current_energy = h[current_note] + self.sum_energy_in_context(
                J, context, current_note
            )
            proposed_note = random.choice(
                [elt for elt in range(self.voc_size) if elt != current_note]
            )
            proposed_energy = h[proposed_note] + self.sum_energy_in_context(
                J, context, proposed_note
            )
            acceptance_ratio = min(1, np.exp(proposed_energy - current_energy))
            if random.random() < acceptance_ratio:
                index_sequence[idx] = proposed_note
        # build sequence of notes
        result = [self.idx_to_note[i] for i in index_sequence]
        return result
