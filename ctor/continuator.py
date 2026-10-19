"""Default MIDI Continuator facade.

`Continuator2` is the stable high-level entry point. It now uses the
VO-Regular-BP backend. Historical engines are preserved in the archive tag.
"""

from ctor.vo_regular_bp import VORegularBPContinuator


class Continuator2(VORegularBPContinuator):
    """Compatibility name for the default VO-Regular-BP MIDI Continuator."""


__all__ = ["Continuator2", "VORegularBPContinuator"]
