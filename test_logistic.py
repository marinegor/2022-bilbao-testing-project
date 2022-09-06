import pytest
import math
import numpy as np
from logistic import logistic_map, iterate_f

SEED = np.random.randint(1, 2**31)

@pytest.fixture
def random_state():
	random_state = np.random.RandomState(SEED)
	print(f"{SEED=}")
	return random_state

@pytest.mark.smoke
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

@pytest.mark.smoke
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


@pytest.mark.slow
def test_logistics_fuzzing(random_state):
	num_it = 100
	n_trials = 5_000
	r = 1.5
	for _ in range(n_trials):
		x0 = random_state.random()
		last_value = iterate_f(r=r, x=x0, it=num_it)[-1]
		assert math.isclose(last_value, 1/3)


@pytest.mark.slow
@pytest.mark.parametrize(
	"num_iterations, num_trials, r",
	[
		(100_000, 100, 3.8),
	]
)
def test_chaotic(num_iterations, num_trials, r, random_state):
	for _ in range(num_trials):
		x = random_state.random()
		values = iterate_f(r=r, x=x, it=num_iterations)

		# check that are bound
		assert all(np.array(values) <= 1) and all(0 <= np.array(values))

		# chech that are aperiodic
		last_values = values[-10_000:]
		assert len(set(last_values)) == len(last_values)

