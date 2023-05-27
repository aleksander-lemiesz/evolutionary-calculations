from abc import abstractmethod
from functools import partial
from random import uniform, sample

from app.SO import SO
from app.utils import bounce, flatten


class MultiSO(SO):

    def __init__(self, population, dimension, opt_function, no_swarms, **kwargs):
        super().__init__(population, dimension, opt_function, **kwargs)
        self.particles = []
        self.no_swarms = no_swarms

        # feed bounce function with constant x_range to speed up computing
        self.bounce = partial(bounce, x_range=self.opt_fun.x_range)

        self.reset()

    def reset(self):
        super().reset()

        # initialises particles splitting them into swarms
        # e.g. [[swarm 0], [swarm 1], ...]; swarms may not be equal in size
        self.particles = [[{
            # random velocity at start
            'v': [uniform(*self.v_init_range) for _ in range(self.dimensions)],
            'x': [uniform(*self.opt_fun.x_range) for _ in range(self.dimensions)]
        } for _ in range(swarm, self.population, self.no_swarms)]
            for swarm in range(self.no_swarms)]

        # global max
        self.best_global = self.particles[0][0]['x'].copy()

    def get_particle(self, swarm_idx, particle_idx):
        return self.particles[swarm_idx][particle_idx]

    def select_best(self):
        flattened = flatten(self.particles)
        computed = {i: self.opt_fun(p['x']) for i, p in enumerate(flattened)}
        # find particle with minimal value of optimisation function
        best_particle_idx = min(computed, key=computed.get)
        self.y = computed[best_particle_idx]
        self.best_global = flattened[best_particle_idx]['x']

    def shuffle(self):
        self.particles = [sample(self.particles[i], k=len(self.particles[i]))
                          for i in range(self.no_swarms)]

    @abstractmethod
    def step(self) -> float:
        pass

    def evaluate(self, iterations: int = None, *args, **kwargs):
        return super().evaluate(self.step, iterations)
