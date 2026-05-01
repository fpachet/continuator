"""Compatibility module alias for the classic MIDI Continuator facade."""

import sys

from ctor.classic import continuator as _impl

sys.modules[__name__] = _impl
