import math


def fun_switcher(fun_id):
    return {
        'f1': {
            'fun': f1,
            'dimension_constraints': (2, 100),
            'x_range': (-100, 100),
            'accuracy': 0.0001
        },
        'f2': {
            'fun': f2,
            'dimension_constraints': (2, 100),
            'x_range': (-100, 100),
            'accuracy': 0.0001
        },
        'f5': {
            'fun': f5,
            'dimension_constraints': (2, 100),
            'x_range': (-5.12, 5.12),
            'accuracy': 30
        },
        'f7': {
            'fun': f7,
            'dimension_constraints': (2, 2),
            'x_range': (-10, 10),
            'accuracy': 0.000001
        },
        'f10': {
            'fun': f10,
            'dimension_constraints': (2, 100),
            'x_range': (-10, 10),
            'accuracy': 0.000001
        },
        'f11': {
            'fun': f11,
            'dimension_constraints': (2, 100),
            'x_range': (-10, 10),
            'accuracy': 0.001
        },
        'f12': {
            'fun': f12,
            'dimension_constraints': (2, 2),
            'x_range': (-100, 100),
            'accuracy': 0.00001
        },
        'f3': {
            'fun': f3,
            'dimension_constraints': (2, 100),
            'x_range': (-100, 100),
            'accuracy': 0.00001
        },
        'f17': {
            'fun': f17,
            'dimension_constraints': (2, 100),
            'x_range': (-100, 100),
            'accuracy': 0.00001
        },
        'f21': {
            'fun': f21,
            'dimension_constraints': (2, 100),
            'x_range': (-100, 100),
            'accuracy': 0.00001
        },
        'f24': {
            'fun': f24,
            'dimension_constraints': (2, 100),
            'x_range': (-65, 65),
            'accuracy': 0.00001
        }
    }.get(fun_id, 'f1')


def f1(vector_x):
    return sum(x * x for x in vector_x)


def f2(vector_x):
    sum_value = 0
    for i, x in enumerate(vector_x):
        x_i = x - i - 1
        sum_value += x_i * x_i
    return sum_value


def f5(vector_x):
    sum_value = 0
    for x in vector_x:
        sum_value += x * x - 10 * math.cos(2 * math.pi * x) + 10
    return sum_value


def f7(vector_x):
    x1 = vector_x[0]
    x2 = vector_x[1]
    x1_pi_2 = (-x1 - math.pi) * (-x1 - math.pi)
    x2_pi_2 = (-x2 - math.pi) * (-x2 - math.pi)
    return -1 * math.cos(x1) * math.cos(x2) * math.exp(x1_pi_2 - x2_pi_2)


def f10(vector_x):
    abs_sum = 0
    product = 1
    for x in vector_x:
        abs_sum += abs(x) * abs(x)
        product *= abs(x)
    return abs_sum + product


def f11(vector_x):
    _sum = sum(map(
        lambda i, xi: xi * i * 0.5,
        range(1, len(vector_x) + 1),
        vector_x
    ))
    return -sum(vector_x) + (_sum2 := _sum * _sum) + _sum2 * _sum2


def f12(vector_x):
    x1_2 = vector_x[0] * vector_x[0]
    x2_2 = vector_x[1] * vector_x[1]
    meter = math.pow(math.sin(math.sqrt(x1_2 + x2_2)), 2) - 0.5
    denominator = math.pow(1 + 0.001 * (x1_2 + x2_2), 2)
    return 0.5 + meter / denominator


def f3(vector_x):
    sum_value = 0
    for i in range(len(vector_x)):
        little_sum = 0
        for j in range(i):
            little_sum += vector_x[j]
        sum_value += little_sum ** 2
    return sum_value


def f17(vector_x):
    sum1 = 0
    sum2 = 0
    for i in range(len(vector_x)):
        sum1 += (vector_x[i] - 1)**2
        sum2 += vector_x[i]**2 - 1
    return sum1 - sum2


def f21(vector_x):
    sum_val= 0
    for i in range(len(vector_x)):
        sum_val += vector_x[i]**2
    return vector_x[1] + 10**6 * sum_val


def f24(vector_x):
    sum1 = 0
    for i in range(len(vector_x)):
        sum2 = 0
        for j in range(len(vector_x)):
            sum2 += vector_x[i]**2
        sum1 += sum2
    return sum1


class OptimizationFunction:
    def __init__(self, opt_function_id: str, **kwargs):
        temp_function = fun_switcher(opt_function_id)
        self.opt_function = temp_function['fun']
        self.dimension_constraints = temp_function['dimension_constraints']
        self.x_range = kwargs.get('x_range', temp_function['x_range'])
        self.accuracy = kwargs.get('accuracy', temp_function['accuracy'])

    def __call__(self, vector_x, *args, **kwargs):
        return self.opt_function(vector_x)
