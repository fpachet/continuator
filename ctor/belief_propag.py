"""
Public infeasibility exception shared by MIDI and symbolic generation.

This module defines no solver. The import and exception name remain part
of the maintained API; inference runs in the external vo_regular_bp backend.
"""


class NoSolutionErrorInBP(Exception):
    """Raised by public APIs when constraints leave no feasible sequence."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message
