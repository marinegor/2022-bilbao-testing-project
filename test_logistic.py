import pytest
import math
from logistic import logistic_map, iterate_f

@pytest.mark.parametrize(
    "x, r, answ",
    [
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.75, 1.7, 0.31875)
    ]
)
def test_logistic_map_values(x, r, answ):
    assert math.isclose(logistic_map(x=x, r=r), answ)

@pytest.mark.parametrize(
    "x, r, it, answ",
    [
        (0.1, 2.2, 1, [0.198]),
        (0.2, 3.4, 4, [0.544, 0.843418, 0.449019, 0.841163]),
        (0.75, 1.7, 2, [0.31875, 0.369152]),
    ]
)
def test_iterate_f(x, r, it, answ):
    our_answ = iterate_f(x=x, r=r, it=it)
    for our_value, given_value in zip(our_answ, answ):
        assert math.isclose(our_value, given_value, rel_tol=1e-6)

