from collections import OrderedDict
from random import uniform, choice

from app.MultiSO import MultiSO
from app.utils import shuffle, grouped


def validate_cso(init):
    """
    Validates CSO inputs in init
    """

    def wrapper(self, population, dimension, opt_function, no_swarms, **kwargs):
        if no_swarms < 2:
            raise Exception(f'Number of swarms should be greater than 2, '
                            f'got {no_swarms}')
        elif population < 2 * no_swarms:
            raise Exception(f'Population should be 2 times greater than number of swarms, '
                            f'got pop: {population}, swarms: {no_swarms}')
        return init(self, population, dimension, opt_function, no_swarms, **kwargs)

    return wrapper


class CSO(MultiSO):

    @validate_cso
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
            swarm_avg_pos = self.calculate_avg_pos(swarm)

            # select only one random winner particle for each swarm
            winners.append(choice(
                # take every 2 particles from swarm only once and get winners
                [self.tournament([(i, p), (i, p + 1)], swarm_avg_pos)
                 for p in range(len(swarm) % 2, len(swarm), 2)]
            ))
        return shuffle(winners)

    def stage_two(self, winners):
        winners_avg_pos = self.calculate_avg_pos(
            [self.get_particle(*w) for w in winners]
        )

        # take every 2 particles from swarm only once
        for bests in grouped(winners, 2):
            self.tournament(bests, winners_avg_pos)

    def tournament(self, idx_particles: list, avg_swarm_pos: list) -> tuple:
        # sort ascending indexes by calculated function value
        ordered = OrderedDict(sorted(
            {i: self.get_particle(*i) for i in idx_particles}.items(),
            key=lambda kv: self.opt_fun(kv[1]['x'])
        ))
        # assign particles to w and los (these are not copies!)
        w, los = ordered.values()

        def count_xv(x_w, x_l, v_l, avg_pos):
            v_l = uniform(0, 1) * v_l \
                  + uniform(0, 1) * (x_w - x_l) \
                  + 0.3 * uniform(0, 1) * (avg_pos - x_l)
            return v_l + x_l, v_l

        # compute new position and v for loser
        xv = list(map(count_xv, w['x'], los['x'], los['v'], avg_swarm_pos))
        # rotate matrix to extract x_l and v_l in vectors
        vector_x_l, los['v'] = list(zip(*xv))

        # perform map to get full vector of bounced positions and save them
        los['x'] = list(map(self.bounce, vector_x_l))

        # return indexes of winner
        return next(iter(ordered))

    def calculate_avg_pos(self, swarm) -> list:
        return [sum(p['x'][d] for p in swarm) / len(swarm)
                for d in range(self.dimensions)]
