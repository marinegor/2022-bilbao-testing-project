import pytest
import math
from logistic import logistic_map

@pytest.mark.parametrize(
    "x, r, answ",
    [
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.75, 1.7, 0.31875)
    ]
)
def test_logistic_map_values(x, r, answ):
    # assert logistic_map(x=x, r=r) == answ
    assert math.isclose(logistic_map(x=x, r=r), answ)



# @decorator
# def func(x):
#     ...
# 
# # the same as 
# 
# def func(x):
#     ...
# 
# func = decorator(func)
