from collections import OrderedDict
from random import uniform, choice

from app.MultiSO import MultiSO
from app.utils import shuffle, grouped


def validate_lcso(init):
    """
    Validates LCSO inputs in init
    """

    def wrapper(self, population, dimension, opt_function, no_swarms, **kwargs):
        if no_swarms < 3:
            raise Exception(f'Number of swarms should be greater than 2, '
                            f'got {no_swarms}')
        elif population < 3 * no_swarms:
            raise Exception(f'Population should be 3 times greater than number of swarms, '
                            f'got pop: {population}, swarms: {no_swarms}')
        return init(self, population, dimension, opt_function, no_swarms, **kwargs)

    return wrapper


class LCSO(MultiSO):

    @validate_lcso
    def __init__(self, population, dimension, opt_function, no_swarms, **kwargs):
        super().__init__(population, dimension, opt_function, no_swarms, **kwargs)

    def step(self) -> float:
        self.shuffle()
        selected_winners = self.stage_one()
        self.stage_two(selected_winners)
        self.select_best()
        return self.y

    def stage_one(self):
        winners = []
        for i, swarm in enumerate(self.particles):
            # select only one random winner particle for each swarm
            winners.append(choice(
                # take every 3 particles from swarm only once and get winners
                [self.tournament([(i, p), (i, p + 1), (i, p + 2)])
                 for p in range(len(swarm) % 3, len(swarm), 3)]
            ))
        return shuffle(winners)

    def stage_two(self, winners):
        # take every 3 particles from swarm only once
        for bests in grouped(winners, 3):
            self.tournament(bests)

    def tournament(self, idx_particles: list) -> tuple:
        # sort ascending indexes by calculated function value
        ordered = OrderedDict(sorted(
            {i: self.get_particle(*i) for i in idx_particles}.items(),
            key=lambda kv: self.opt_fun(kv[1]['x'])
        ))
        # assign particles to w, s and los (these are not copies!)
        w, s, los = ordered.values()

        def count_xv(x_w, x_s, x_l, v_s, v_l):
            v_s = uniform(0, 1) * v_s + uniform(0, 1) * (x_w - x_s)
            v_l = uniform(0, 1) * v_l \
                + uniform(0, 1) * (x_w - x_l) \
                + uniform(0, 1) * (x_s - x_l)
            return v_s + x_s, v_l + x_l, v_s, v_l

        # compute new position and v for s and loser
        xv = list(map(count_xv, w['x'], s['x'], los['x'], s['v'], los['v']))
        # rotate matrix to extract x_s, x_l, v_s and v_l in vectors
        vector_x_s, vector_x_l, vector_v_s, vector_v_l = list(zip(*xv))

        # perform map to get full vector of x
        s['x'] = list(map(self.bounce, vector_x_s))
        los['x'] = list(map(self.bounce, vector_x_l))
        s['v'] = vector_v_s
        los['v'] = vector_v_l

        # return indexes of winner
        return next(iter(ordered))
