"""Euler-style challenge: Minimal solutions to Pell's equation x^2 - D*y^2 = 1."""


def solve_pell(D: int) -> tuple[int, int]:
    """Return the minimal positive integer solution (x, y) to x^2 - D * y^2 = 1.

    Raise ValueError if D < 2 or D is a perfect square.
    Must use exact integer arithmetic (e.g., continued fractions of sqrt(D)).
    """
    raise NotImplementedError


def max_x_pell_d(limit: int) -> int:
    """Return the non-square integer D <= limit with the largest minimal x."""
    raise NotImplementedError
