import math
from typing import List


def find_sum_of_squares(l: int, r: int) -> List[int]:
    """
    Return all integers n in [l, r] that can be written as a^2 + b^2
    for some integers a, b >= 0. Results are unique and sorted.

    Parameters
    ----------
    l : int
        Lower bound (inclusive), must be >= 0.
    r : int
        Upper bound (inclusive), must be >= 0.

    Returns
    -------
    values : List[int]
        Sorted list of integers n in [l, r] such that n = a^2 + b^2.

    Notes
    -----
    - We enumerate a, b >= 0 with b >= a to avoid symmetric duplicates.
    - This function is intended for generating "critical radii squared"
      in discrete VP-circle exact search.
    """
    if not isinstance(l, int) or not isinstance(r, int):
        raise TypeError("l and r must be integers")
    if l < 0 or r < 0:
        raise ValueError("l and r must be >= 0")
    if l > r:
        return []

    out = set()
    a_max = int(math.isqrt(r))

    for a in range(a_max + 1):
        a2 = a * a
        # b^2 <= r - a^2
        b_max = int(math.isqrt(r - a2))
        for b in range(a, b_max + 1):
            n = a2 + b * b
            if n < l:
                continue
            out.add(n)

    return sorted(out)
