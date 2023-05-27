import math
from collections.abc import Callable

from app.optimization_functions import OptimizationFunction

MAX_FLOAT = float('inf')


class SO:
    def __init__(self, population, dimension, opt_function, **kwargs):
        assert isinstance(opt_function, OptimizationFunction)

        if opt_function.dimension_constraints[0] > dimension \
                or opt_function.dimension_constraints[1] < dimension:
            raise Exception(f'Given dimension is out of boundaries: '
                            f'{dimension} not in {opt_function.dimension_constraints}')

        self.opt_fun = opt_function
        self.dimensions = dimension
        self.population = population

        # best x-es with best y
        self.best_global = []
        self.y = MAX_FLOAT

        velocity_magnitude = kwargs.get('velocity_magnitude', 0.0)
        # (x_range difference * -magnitude, x_range difference * magnitude)
        self.v_init_range = list(map(
            lambda m: (self.opt_fun.x_range[1] - self.opt_fun.x_range[0]) * m,
            (-velocity_magnitude, velocity_magnitude)
        ))

        self.logs = {}

    def reset(self):
        self.y = MAX_FLOAT
        self.logs = {}

    def evaluate(
            self,
            step_function: Callable[[], float],
            iterations: int = None
    ):
        logs_y = []
        # accuracy mode
        if iterations is None:
            for i in range(10000):
                y, self.y = self.y, step_function()
                logs_y.append(self.y)

                # 10 last solutions are equal => break
                if 0 < math.fabs(self.y - y) <= self.opt_fun.accuracy \
                        or len(logs_y) > 50 \
                        and all(logs_y[-1] == log_y for log_y in logs_y[-50:]):
                    self.logs['iterations'] = i
                    break
            else:
                self.logs['iterations'] = 10000

        # iteration mode
        elif isinstance(iterations, int) and iterations > 0:
            for i in range(iterations):
                y = step_function()
                logs_y.append(y)
            self.logs['iterations'] = iterations
        else:
            raise Exception(f'Iterations should be integer or left empty, got {iterations}')

        self.logs['y'] = tuple(logs_y)
        return self.logs['y'][-1]
