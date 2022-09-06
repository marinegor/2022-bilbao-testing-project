def logistic_map(x: float, r: float) -> float:
    return r*x*(1-x)

def iterate_f(it: int, x, r) -> list[float]:
    rv = []
    for _ in range(it):
        x = logistic_map(x, r)
        rv.append(x)
    return rv

