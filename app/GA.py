import pygad

from app.optimization_functions import OptimizationFunction


class GA:

    def __init__(self, dimension, opt_function, iterations, **kwargs):
        assert isinstance(opt_function, OptimizationFunction)

        if opt_function.dimension_constraints[0] > dimension \
                or opt_function.dimension_constraints[1] < dimension:
            raise Exception(f'Given dimension is out of boundaries: '
                            f'{dimension} not in {opt_function.dimension_constraints}')

        def fitness_func(solution, _):
            output = opt_function(solution)
            fitness = -output
            return fitness

        self.ga_instance = pygad.GA(
            num_generations=iterations,
            num_parents_mating=kwargs.get('num_parents_mating', 2),
            num_genes=dimension,
            sol_per_pop=kwargs.get('sol_per_pop', 50),
            init_range_low=opt_function.x_range[0],
            init_range_high=opt_function.x_range[1],
            parent_selection_type=kwargs.get('parent_selection_type', 'sss'),
            keep_parents=kwargs.get('keep_parents', 1),
            crossover_type=kwargs.get('crossover_type', 'single_point'),
            mutation_type=kwargs.get('mutation_type', 'random'),
            mutation_percent_genes=kwargs.get('mutation_percent_genes', 5),
            fitness_func=fitness_func,
            mutation_by_replacement=True,
            random_mutation_min_val=opt_function.x_range[0],
            random_mutation_max_val=opt_function.x_range[1],
            **kwargs
        )
        self.opt_function = opt_function

    def run(self):
        self.ga_instance.run()

    def best_solution(self):
        solution, _, _ = self.ga_instance.best_solution()
        return solution, self.opt_function(solution)
