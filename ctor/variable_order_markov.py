"""Deprecated generation-only adapter; the historical engine is archived."""

import warnings

from ctor.sequence_model import SequenceModel


class Variable_order_Markov(SequenceModel):
    """Generation subset only; classic counters, decay and marginals retired."""

    def __init__(self, sequence_of_stuff=None, vp_lambda=None, kmax=4, *, seed=0):
        warnings.warn(
            "Variable_order_Markov now adapts VO-Regular-BP generation only; "
            "use ctor.sequence_model.SequenceModel. Classic counters/decay/marginals were archived.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(kmax=kmax, viewpoint_fn=vp_lambda, seed=seed)
        if sequence_of_stuff is not None:
            self.learn_sequence(sequence_of_stuff)
