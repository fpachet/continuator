"""Compatibility module alias for the classic variable-order Markov model."""

import sys

from ctor.classic import variable_order_markov as _impl

sys.modules[__name__] = _impl
