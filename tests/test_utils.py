from app.utils import bounce
from functools import partial


def test_bounce():
    vector_x = list(range(-10, 10))
    boundaries = (-1, 3)
    bounce_with_boundaries = partial(bounce, x_range=boundaries)
    for x in vector_x:
        assert boundaries[0] <= bounce_with_boundaries(x) <= boundaries[1]
