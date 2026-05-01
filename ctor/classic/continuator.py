"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

import numpy as np

from ctor.midi import MidiContinuatorBase
from ctor.classic.variable_order_markov import Variable_order_Markov

"""
- Split the music Continuator class from a generic Variable_Order_Markov, usable for any type of sequence (e.g. words).
- Implementation of Continuator is different from original, to enable experiments with belief propagation and skips.
- Representation of contexts of size 1 to K and their continuations with dictionaries. Trees/oracles are useless here.
- Contexts are tuples of viewpoints AND continuations are viewpoints (see get_viewpoint()) (Unlike in the original)
- Realizations are kept separately for each vp and reused during sampling. They are represented as addresses, i.e. tuple (index_of_melody, index_in_melody)
- Sampling attempts to avoid too long repetitions (a kind of max-order) by avoiding singletons when it can
- Sampling is performed both by belief propagation (1st order) and by variable-order and combined
- Realization of viewpoints is performed with dynamic programming, à la HMM
- Representation of polyphony is different from original Continuator. Clusters are not considered, only notes.
They have a "status" describing how they were played originally, which is preserved at sampling. This enables more creativity for chords.
- TODO: retrain periodically with computed viewpoints (bin quartiles for durations and velocity)
- TODO: audio synthesis with Dawdreamer
- TODO: add database storage of real time performances
- TODO: data augmentation with inversions, negative harmony, etc.
- TODO: rhythm transfer for data augmentation/control
- TODO: server with js client, or huggingface solution or github page with python2js
- TODO: use fine-tuning of transformers
"""

class ClassicContinuator(MidiContinuatorBase):
    """Classic MIDI Continuator facade backed by `Variable_order_Markov`."""

    def __init__(self, midi_file: object = None, kmax: int = 4, transposition: bool = False) -> None:
        # self.vom = Variable_order_Markov(None, self.get_viewpoint, kmax)
        self.vom = Variable_order_Markov(
            sequence_of_stuff=None,
            vp_lambda=self.get_viewpoint,  # identity viewpoint
            kmax=kmax,
            decay_fast_half_life=10,  # forget half the influence every ~10 events
            decay_slow_half_life=80,  # for 'middle' band if you test it later
            seed=0
        )
        # self.vom.set_period_mode("late")  # 'late' uses recent (fast-decayed) counts

        self.initialize_midi_state(transposition)
        if midi_file is not None:
            self.learn_file(midi_file, transposition)


    def quantile_bins(self, values, N):
        """
        Compute bin edges that split the input values into N bins
        with approximately equal number of points (quantiles).

        Args:
            values: list or array of numerical values (e.g., durations or velocities)
            N: number of desired bins

        Returns:
            bin_edges: list of N+1 edges that define the bin intervals
        """
        values = np.array(values)
        quantiles = np.linspace(0, 1, N + 1)
        bin_edges = np.quantile(values, quantiles)
        return bin_edges.tolist()


    def get_all_input_durations(self):
        all_durations = []
        for note_seq in self.vom.input_sequences:
            all_durations = all_durations + [n.duration for n in note_seq]
        return all_durations

    def compute_viewpoints(self, note_sequence):
        """
        Placeholder for future adaptive viewpoint binning.

        This currently has no side effects. It gathers durations that could be
        used later for quantile-based duration or velocity viewpoints.
        """
        all_durations = self.get_all_input_durations()
        all_durations = all_durations + [n.duration for n in note_sequence]
        all_durations.sort()
        # print(self.vom.all_unique_viewpoints)
        # print(self.quantile_bins(all_durations, 2))

    def learn_phrase(self, note_sequence, transposition):
        # TODO: update adaptive viewpoint quantizers here when duration,
        # velocity, or articulation bins become active.
        # self.compute_viewpoints(note_sequence)
        if len(note_sequence) == 0:
            return
        if self.forget_past and self.keep_last_n_melodies <= len(self.vom.input_sequences):
            self.clear_first_n_phrases(1 + len(self.vom.input_sequences) - self.keep_last_n_melodies)
        # all_pitches = [note.pitch for note in note_sequence]
        # print(f"number of different pitches in train: {len(Counter(all_pitches))}")
        # print(f"min pitch: {min(all_pitches)}, max pitch: {max(all_pitches)}")
        # learns, possibly in 12 transpositions
        trange = range(0, 1)
        if transposition:
            trange = range(-6, 6, 1)
        for t in trange:
            transposed = self.transpose_notes(note_sequence, t)
            # learns one more sequence
            self.vom.learn_sequence(transposed)

    def get_start_vp(self):
        return self.vom.start_padding

    def get_end_vp(self):
        return self.vom.end_padding

    def sample_sequence(
            self,
            prefix=None,
            length=50,
            constraints=None,
            start_vp=None,
            relax_prefix_on_fail=True,
            relax_pos0_on_fail=True,
            raise_on_fail=False,
    ):
        """
        :param length:
        :type constraints: dict
        """
        return self.vom.sample_sequence(
            length,
            prefix=prefix,
            constraints=constraints,
            start_vp=start_vp,
            relax_prefix_on_fail=relax_prefix_on_fail,
            relax_pos0_on_fail=relax_pos0_on_fail,
            raise_on_fail=raise_on_fail,
        )

    def continue_sequence(
            self,
            prefix,
            length=50,
            constraints=None,
            relax_prefix_on_fail=True,
            relax_pos0_on_fail=True,
            raise_on_fail=False,
    ):
        return self.vom.continue_sequence(
            prefix,
            length=length,
            constraints=constraints,
            relax_prefix_on_fail=relax_prefix_on_fail,
            relax_pos0_on_fail=relax_pos0_on_fail,
            raise_on_fail=raise_on_fail,
        )

    def continue_until_end(self, prefix=None, min_length=1, max_length=64, end_vp=None):
        return self.vom.continue_until_end(
            prefix=prefix,
            min_length=min_length,
            max_length=max_length,
            end_vp=end_vp,
        )

    def sample_sequence_0(self, length=50, constraints=None):
        """
        :param length:
        :type constraints: dict
        """
        return self.vom.sample_zero_order(length, constraints=constraints)


class Continuator2(ClassicContinuator):
    """Compatibility name for the classic MIDI Continuator."""
