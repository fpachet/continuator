"""
Compatibility exceptions for historical belief-propagation callers.

The old generic graph implementation was removed from the current core;
constrained sequence inference now uses the sparse chain solver in
ctor.chain_solver.
"""


class NoSolutionErrorInBP(Exception):
    """Raised by public APIs when constraints leave no feasible sequence."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message
