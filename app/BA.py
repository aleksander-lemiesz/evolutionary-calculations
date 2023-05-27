from functools import partial
from random import uniform
from cmath import exp

from app.SO import SO
from app.utils import bounce, levy_flight


class BA(SO):

    def __init__(self, population, dimension, opt_function, levy=False, **kwargs):
        super().__init__(population, dimension, opt_function)

        self.bats = {}
        self.t = 0

        # feed bounce function with constant x_range to speed up computing
        self.bounce = partial(bounce, x_range=self.opt_fun.x_range)
        self.levy_or_not = lambda v: levy_flight(v) if levy else v

        self.f_range = kwargs.get('freq_range', (0, 2))

        self.mod_r = kwargs.get('mod_r', 0.8)  # pulse rate modifier | mod_r > 0
        self.mod_A = kwargs.get('mod_A', 0.9)  # loudness modifier | 0 < mod_a < 1

        self.reset()

    def step(self) -> float:
        self.t += 1
        [self.rate_bat(bat) for bat in self.bats]

        for bat in self.bats:
            # calculate vectors of velocity and position
            bat['v'] = list(map(
                lambda v, x: v + self.count_freq() * (x - self.best_global),
                bat['v'], bat['x']
            ))
            bat['x'] = list(map(
                lambda v, x: self.bounce(x + self.levy_or_not(v)),
                bat['v'], bat['x']
            ))

        for bat in self.bats:
            if uniform(0, 1) > bat['r']:
                self.echolocation(self.best_global)
            else:
                self.echolocation(bat)
        return self.y

    def rate_bat(self, bat: dict):
        bat['y'] = self.opt_fun(bat['x'])
        if bat['y'] < self.y:
            self.y = bat['y'].copy()
            self.best_global = bat

    def echolocation(self, bat: dict):
        pos = list(map(
            lambda x: x + uniform(-1, 1) * self.count_avg_loudness(),
            bat['x']
        ))
        y = self.opt_fun(pos)
        if y < bat['y'] and uniform(0, 1) < bat['A']:
            bat['y'], bat['x'] = y, pos
            bat['A'] = bat['A'] * self.mod_A
            bat['r'] = bat['r'] * (1 - exp(-self.mod_r * self.t))

    def count_avg_loudness(self):
        return sum(bat['A'] for bat in self.bats) / len(self.bats)

    def count_freq(self):
        return self.f_range[0] + uniform(0, 1) * (self.f_range[1] - self.f_range[0])

    def evaluate(self, iterations=None, *args, **kwargs):
        return super().evaluate(self.step, iterations)

    def reset(self):
        super().reset()

        self.bats = [{
            'v': [0] * self.dimensions,
            # actual position of particle in dimension
            'x': [uniform(*self.opt_fun.x_range) for _ in range(self.dimensions)],
            'A': uniform(1, 2),  # loudness
            'r': 0.5  # pulse rate
        } for _ in range(self.population)]

        # best bat
        self.t = 0
