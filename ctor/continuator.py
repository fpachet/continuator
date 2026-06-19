"""Default MIDI Continuator facade.

`Continuator2` is the stable high-level entry point. It now uses the
VO-Regular-BP backend by default, while `ClassicContinuator` remains available
for callers that explicitly need the classic engine.
"""

from ctor.classic import ClassicContinuator
from ctor.vo_regular_bp import VORegularBPContinuator


class Continuator2(VORegularBPContinuator):
    """Compatibility name for the default VO-Regular-BP MIDI Continuator."""


__all__ = ["ClassicContinuator", "Continuator2", "VORegularBPContinuator"]
