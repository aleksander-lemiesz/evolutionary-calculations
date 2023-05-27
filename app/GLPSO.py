from random import uniform, random
import random

from app.SO import SO
from app.utils import bounce, levy_flight


def call_w(w):
    """if 'w' looks like [*args] call uniform with args"""
    if isinstance(w, (tuple, list)):
        return uniform(*w)
    return w


class GLPSO(SO):

    def __init__(self, population, dimension, opt_function, pm, levy=False, **kwargs):
        super().__init__(population, dimension, opt_function)

        self.particles = {}
        self.exemplars = {}
        self.offsprings = {}

        self.w_v = kwargs.get('w_v', 0.729)
        self.w_l = kwargs.get('w_l', 1.494)
        self.w_g = kwargs.get('w_g', 1.494)
        self.pm = pm

        self.levy_or_not = lambda v: levy_flight(v) if levy else v

        self.reset()

    def step(self) -> float:
        i = 0
        for pn in self.particles:
            for d in range(self.dimensions):
                # Losowo wybierz cząstkę
                p2 = self.particles[random.randint(0, len(self.particles)-1)]
                # Krzyżowanie
                if self.opt_fun(pn['best_local']) < self.opt_fun(p2['best_local']):
                    rd = uniform(0, 1)
                    self.offsprings[i]['x'][d] = rd * pn['best_local'][d] + (1 - rd) * self.best_global[d]
                else:
                    self.offsprings[i]['x'][d] = p2['best_local'][d]
                # Mutacja
            for d in range(self.dimensions):
                if uniform(0, 1) < self.pm:
                    self.offsprings[i]['x'][d] = uniform(self.opt_fun.x_range[0], self.opt_fun.x_range[1])
                    self.offsprings[i]['v'][d] = uniform(-10, 10)
                # Oblicz exemplar
            for d in range(self.dimensions):
                self.exemplars[i]['x'][d] = self.calculate_exemplar(i, d)
                # Selekcja
            if self.opt_fun(self.offsprings[i]['x']) < self.opt_fun(self.exemplars[i]['x']):
                self.exemplars[i]['x'] = self.offsprings[i]['x']
                # Update cząstek
            for d in range(self.dimensions):
                # Calculate new velocity
                v = call_w(self.w_v) * pn['v'][d] \
                    + (call_w(self.w_l) + call_w(self.w_g)) / 2 * uniform(0, 1) \
                    * (self.exemplars[i]['x'][d] - pn['x'][d])
                pn['v'][d] = self.levy_or_not(v)

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

    def calculate_exemplar(self, i, d):
        c1 = call_w(self.w_l)
        c2 = call_w(self.w_g)
        r1 = uniform(0, 1)
        r2 = uniform(0, 1)
        return (c1 * r1 * self.particles[i]['best_local'][d] + c2 * r2 * self.best_global[d]) / (c1 * r1 + c2 * r2)

    def evaluate(self, iterations=None, *args, **kwargs):
        return super().evaluate(self.step, iterations)

    def reset(self):
        super().reset()

        self.particles = [{
            'v': [0] * self.dimensions,
            # actual position of particle in dimension
            'x': [uniform(*self.opt_fun.x_range) for _ in range(self.dimensions)]
        } for _ in range(self.population)]

        self.offsprings = [{
            'v': [0] * self.dimensions,
            # actual position of particle in dimension
            'x': [uniform(*self.opt_fun.x_range) for _ in range(self.dimensions)]
        } for _ in range(self.population)]

        self.exemplars = [{
            'v': [0] * self.dimensions,
            # actual position of particle in dimension
            'x': [uniform(*self.opt_fun.x_range) for _ in range(self.dimensions)]
        } for _ in range(self.population)]

        # local max
        for p in self.particles:
            p['best_local'] = p['x'].copy()
        # global max
        self.best_global = self.particles[0]['x'].copy()
