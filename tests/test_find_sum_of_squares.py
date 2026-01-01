import pytest

from vp_circles.find_sum_of_squares import find_sum_of_squares


def test_sum_of_two_squares_basic_range_0_25():
    """
    Basic correctness on a small range [0, 25].

    Expected behavior:
    - Return exactly all n in [0, 25] representable as a^2 + b^2 (a, b >= 0).
    - Results must be sorted and contain no duplicates.
    """
    got = find_sum_of_squares(0, 25)
    expected = [0, 1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25]
    assert got == expected
    assert got == sorted(got)
    assert len(got) == len(set(got))


def test_sum_of_two_squares_range_1_10():
    """
    Range query that excludes zero.

    Expected behavior:
    - Return all and only the representable values within [1, 10].
    - Results must be sorted.
    """
    got = find_sum_of_squares(1, 10)
    assert got == [1, 2, 4, 5, 8, 9, 10]
    assert got == sorted(got)


def test_sum_of_two_squares_single_value_true():
    """
    Single-point query where the value is representable.

    Expected behavior:
    - If l == r and the number is representable, return [l].
    """
    assert find_sum_of_squares(50, 50) == [50]


def test_sum_of_two_squares_single_value_false():
    """
    Single-point query where the value is not representable.

    Expected behavior:
    - If l == r and the number is not representable, return [].
    """
    assert find_sum_of_squares(3, 3) == []


def test_sum_of_two_squares_empty_when_l_greater_than_r():
    """
    Empty interval handling.

    Expected behavior:
    - If l > r, return an empty list.
    """
    assert find_sum_of_squares(10, 9) == []


def test_sum_of_two_squares_sorted_and_unique():
    """
    General invariants on a larger range.

    Expected behavior:
    - Returned values must be strictly unique.
    - Returned values must be sorted in ascending order.
    """
    got = find_sum_of_squares(0, 200)
    assert got == sorted(got)
    assert len(got) == len(set(got))


def test_sum_of_two_squares_zero_inclusion():
    """
    Zero boundary behavior.

    Expected behavior:
    - Include 0 if and only if 0 is inside the query range.
    """
    assert find_sum_of_squares(0, 0) == [0]
    assert find_sum_of_squares(1, 1) == [1]


def test_sum_of_two_squares_type_errors():
    """
    Input type validation.

    Expected behavior:
    - Non-integer inputs must raise TypeError.
    """
    with pytest.raises(TypeError):
        find_sum_of_squares(0.0, 10)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        find_sum_of_squares(0, "10")  # type: ignore[arg-type]


def test_sum_of_two_squares_value_errors_for_negative_inputs():
    """
    Input value validation.

    Expected behavior:
    - Negative bounds must raise ValueError.
    """
    with pytest.raises(ValueError):
        find_sum_of_squares(-1, 10)
    with pytest.raises(ValueError):
        find_sum_of_squares(0, -10)
