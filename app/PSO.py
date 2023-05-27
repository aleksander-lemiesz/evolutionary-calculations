from random import uniform

from app.SO import SO
from app.utils import bounce


def call_w(w):
    """if 'w' looks like [*args] call uniform with args"""
    if isinstance(w, (tuple, list)):
        return uniform(*w)
    return w


class PSO(SO):

    def __init__(self, population, dimension, opt_function, **kwargs):
        super().__init__(population, dimension, opt_function)

        self.particles = {}

        self.w_v = kwargs.get('w_v', 0.729)
        self.w_l = kwargs.get('w_l', 1.494)
        self.w_g = kwargs.get('w_g', 1.494)

        self.reset()

    def step(self) -> float:
        for pn in self.particles:
            for d in range(self.dimensions):
                # Calculate new velocity
                v = call_w(self.w_v) * pn['v'][d] \
                    + call_w(self.w_l) * uniform(0, 1) * (pn['best_local'][d] - pn['x'][d]) \
                    + call_w(self.w_g) * uniform(0, 1) * (self.best_global[d] - pn['x'][d])
                pn['v'][d] = v

                # Check for edge and change position
                pn['x'][d] = bounce(pn['x'][d] + v, self.opt_fun.x_range)

            # Calculate new value
            f_value = self.opt_fun(pn['x'])

            # Check if value is new global minimum
            if f_value < self.y:
                self.best_global = pn['x'].copy()
                self.y = f_value

            # Check if value is new local minimum
            if f_value < self.opt_fun(pn['best_local']):
                pn['best_local'] = pn['x'].copy()

        return self.y

    def alt_step(self) -> float:
        avg_diff = [0] * self.dimensions
        for d in range(self.dimensions):
            for pn in self.particles:
                avg_diff[d] += self.best_global[d] - pn['x'][d]
            avg_diff[d] /= len(self.particles)

        for pn in self.particles:
            for d in range(self.dimensions):
                # Calculate new velocity
                v = call_w(self.w_v) * pn['v'][d] \
                    + call_w(self.w_l) * uniform(0, 1) * (pn['best_local'][d] - pn['x'][d]) \
                    + call_w(self.w_g) * uniform(0, 1) * (avg_diff[d] - pn['x'][d])
                pn['v'][d] = v

                # Check for edge and change position
                pn['x'][d] = bounce(pn['x'][d] + v, self.opt_fun.x_range)

            # Calculate new value
            f_value = self.opt_fun(pn['x'])

            # Check if value is new global minimum
            if f_value < self.y:
                self.best_global = pn['x'].copy()
                self.y = f_value

            # Check if value is new local minimum
            if f_value < self.opt_fun(pn['best_local']):
                pn['best_local'] = pn['x'].copy()

        return self.y

    def evaluate(self, iterations=None, alternative=False, *args, **kwargs):
        if alternative:
            return super().evaluate(self.alt_step, iterations)
        else:
            return super().evaluate(self.step, iterations)

    def reset(self):
        super().reset()

        self.particles = [{
            'v': [0] * self.dimensions,
            # actual position of particle in dimension
            'x': [uniform(*self.opt_fun.x_range) for _ in range(self.dimensions)]
        } for _ in range(self.population)]

        # local max
        for p in self.particles:
            p['best_local'] = p['x'].copy()
        # global max
        self.best_global = self.particles[0]['x'].copy()
