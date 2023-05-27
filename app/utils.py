import random
from functools import reduce
from operator import concat
from random import sample, uniform
from scipy.stats import levy
from math import fabs


def bounce(x, x_range):
    if x > 0:
        x = x % (2 * fabs(x_range[0] - x_range[1]))
    elif x < 0:
        x = (x % (-2 * fabs(x_range[0] - x_range[1])))
    if x < x_range[0]:
        return bounce(x_range[0] + fabs(x_range[0] - x), x_range)
    elif x > x_range[1]:
        return bounce(x_range[1] - fabs(x_range[1] - x), x_range)
    return x


def glue_to_wall(x, x_range):
    if x < x_range[0]:
        return x_range[0]
    elif x > x_range[1]:
        return x_range[1]
    else:
        return x


def shuffle(iterable):
    return sample(iterable, k=len(iterable))


def grouped(iterable, n):
    """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), ..."""
    return zip(*[iter(iterable)] * n)


def flatten(iterable):
    """((s00, s01), (s10, s11), ...) -> (s00, s01, s10, s11, ...)"""
    return reduce(concat, iterable)


def levy_flight(x):
    if uniform(0, 1) > 0.98:
        return x * levy.rvs()
    else:
        return x
