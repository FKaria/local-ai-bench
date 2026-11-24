"""Interval merging utilities."""

from typing import Iterable


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge all overlapping or touching closed intervals [start, end].

    - Input may be unsorted and may be empty.
    - Each interval satisfies start <= end (coordinates may be negative).
    - Touching intervals like (1, 3) and (3, 5) MUST merge into (1, 5).
    - Return a new list of (start, end) tuples sorted in ascending order.
    """
    raise NotImplementedError


def total_covered_length(intervals: Iterable[tuple[int, int]]) -> int:
    """Return the total length covered by the union of intervals.

    For merged intervals [(a1, b1), (a2, b2), ...], return sum(b_i - a_i).
    """
    raise NotImplementedError
