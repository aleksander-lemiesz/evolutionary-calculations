import pytest

from app.optimization_functions import OptimizationFunction


@pytest.fixture
def opt_funct():
    def _make_opt_fun(fun_id, x_range, accuracy):
        return OptimizationFunction(fun_id, x_range=x_range, accuracy=accuracy)
    return _make_opt_fun


@pytest.fixture
def sum_opt_funct(opt_funct):
    f = opt_funct('f1', (-10, 10), 0.1)
    f.opt_function = sum
    return f


@pytest.fixture
def f1_opt_funct(opt_funct):
    return opt_funct('f1', (-100, 100), 0.1)
