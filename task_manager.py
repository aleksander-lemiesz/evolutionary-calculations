import csv

from datetime import datetime
from itertools import zip_longest

from app.GA import GA
from app.GLPSO import GLPSO
from app.PSO import PSO
from app.LCSO import LCSO
from app.CSO import CSO
from app.SO import SO
from app.optimization_functions import OptimizationFunction


def write_csv(filename, headers, csv_data, csv_dir='files'):
    with open(f'{csv_dir}/{filename}', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_data)


class TaskManager:
    def __init__(self, data, user_inputs, inputs_data):
        self.data = data
        self.user_inputs = user_inputs
        self.inputs_data = inputs_data

        # extract global parameters
        # repeats: number of times to repeat the algorithm
        self.repeats = data['settings']['repeats']
        # w_parameters: sets of 'w' variable in dictionary
        self.w_parameters = data['w_parameters']
        # activate_ga: if GA algorithm should be also used for each PSO
        self.activate_ga = data['settings']['activate_ga']
        # hide_prints: hide prints of details and logs
        self.hide_prints = data['settings']['hide_prints']

        # save csv-s for logs
        self.save_csv_summary = data['settings']['save_csv_summary']
        self.save_csv_details = data['settings']['save_csv_details']
        self.save_csv_y_matrix = data['settings']['save_csv_y_matrix']

        # logs
        # averages over repeats
        self.avg_y, self.avg_iterations, self.avg_times = [], [], []
        # y-s for each repeat
        self.y_matrix = []

    def reset_logs(self):
        self.avg_y, self.avg_iterations, self.avg_times = [], [], []
        self.y_matrix = []

    def start_tasks(self):
        # loop over all inputs
        for user_input, input_data \
                in zip(self.user_inputs, self.inputs_data):
            print('=' * 100)
            print(f'{user_input}: {input_data}')

            if user_input.startswith('pso_'):
                self.pso_task(user_input, input_data)
            elif user_input.startswith('lcso_'):
                self.lcso_task(user_input, input_data)
            elif user_input.startswith('cso_'):
                self.cso_task(user_input, input_data)
            elif user_input.startswith('glpso_'):
                self.glpso_task(user_input, input_data)
            elif user_input.startswith('ba_'):
                self.ba_task(user_input, input_data)
            else:
                raise Exception(f'Algorithm for input {user_input} not recognised')

        # save to csv avg_y and avg_iterations for every input in one summary
        if self.save_csv_summary:
            write_csv(
                'summary.csv',
                ('input', 'avg_y', 'avg_iterations', 'avg_times'),
                zip(self.user_inputs, self.avg_y, self.avg_iterations, self.avg_times)
            )

    def so_task(self, so_object: SO, **evaluate_kwargs) -> [[int], [int]]:
        y, iterations, times = [], [], []
        for i in range(self.repeats):
            print(f'===REPEAT {i + 1}===')

            start_time = datetime.now()
            y.append(so_object.evaluate(**evaluate_kwargs))
            run_time = datetime.now() - start_time

            # log y, iterations and time to find solution
            times.append(run_time.seconds * 1000 + run_time.microseconds // 1000)
            iterations.append(so_object.logs['iterations'])
            self.y_matrix.append(so_object.logs['y'])

            if not self.hide_prints:
                print(f'Best solution {so_object.y} for {so_object.best_global}')
                print(so_object.logs)
            so_object.reset()

        self.avg_y.append(sum(y) / self.repeats)
        self.avg_iterations.append(sum(iterations) // self.repeats)
        self.avg_times.append(sum(times) // self.repeats)

        print(f'Solutions : {y}')
        print(f'Iterations: {iterations}')
        print(f'Average best solution        : {self.avg_y[-1]}')
        print(f'Average no iterations        : {self.avg_iterations[-1]}')
        print(f'Average time to find solution: {self.avg_times[-1]} ms')

        return y, iterations, times

    def pso_task(self, user_input, input_data):
        # init classes
        opt_function = OptimizationFunction(input_data['function'])
        pso = PSO(
            input_data['population'],
            input_data['dimension'],
            opt_function,
            **self.w_parameters[input_data['w_set']]
        )

        evaluate_kwargs = {
            'iterations': input_data.get('iterations', None),
            'alternative': input_data.get('alternative', False)
        }

        y, iterations, _ = self.so_task(pso, **evaluate_kwargs)

        if self.activate_ga:
            self.ga_subtask(input_data, opt_function)

        # save to csv y and iterations
        if self.save_csv_details:
            variant = user_input.removeprefix('pso_')
            write_csv(
                f'{user_input}.csv',
                (f'{variant}_solution', f'{variant}_iterations'),
                zip(y, iterations)
            )
        # take only actual repeats (from the last input)
        cur_y_matrix = self.y_matrix[-self.repeats:]
        # save all y-s in every repeat
        if self.save_csv_y_matrix:
            write_csv(
                f'{user_input}_y_matrix.csv',
                range(1, self.repeats + 1),
                # needs to rotate matrix
                zip_longest(*cur_y_matrix[::-1])
            )

    def ga_subtask(self, input_data, opt_function):
        y_ga, iterations_ga = [], []
        for i in range(self.repeats):
            print(f'===REPEAT {i + 1}===')
            ga = GA(
                input_data['dimension'],
                opt_function,
                input_data.get('iterations', 100)
            )
            ga.run()
            solution, solution_fitness = ga.best_solution()
            y_ga.append(solution_fitness)
            iterations_ga.append(input_data.get('iterations', 100))
            print(y_ga)

        if self.save_csv_details:
            write_csv(
                f'ga_{input_data["function"]}.csv',
                (f'ga_{input_data["function"]}_solution',
                 f'ga_{input_data["function"]}_iterations'),
                zip(y_ga, iterations_ga)
            )

    def lcso_task(self, user_input, input_data):
        # init classes
        opt_function = OptimizationFunction(input_data['function'])
        lcso = LCSO(
            input_data['population'],
            input_data['dimension'],
            opt_function,
            input_data['no_swarms'],
            velocity_magnitude=input_data.get('velocity_magnitude', 0.0)
        )

        evaluate_kwargs = {
            'iterations': input_data.get('iterations', None)
        }

        y, iterations, times = self.so_task(lcso, **evaluate_kwargs)

        # save to csv y and iterations
        if self.save_csv_details:
            write_csv(
                f'{user_input}.csv',
                (f'{user_input}_solution',
                 f'{user_input}_iterations',
                 f'{user_input}_runtime'),
                zip(y, iterations, times)
            )
        # take only actual repeats (from the last input)
        cur_y_matrix = self.y_matrix[-self.repeats:]
        # save all y-s in every repeat
        if self.save_csv_y_matrix:
            write_csv(
                f'{user_input}_y_matrix.csv',
                range(1, self.repeats + 1),
                # needs to rotate matrix
                zip_longest(*cur_y_matrix[::-1])
            )

    def cso_task(self, user_input, input_data):
        # init classes
        opt_function = OptimizationFunction(input_data['function'])
        cso = CSO(
            input_data['population'],
            input_data['dimension'],
            opt_function,
            input_data['no_swarms'],
            velocity_magnitude=input_data.get('velocity_magnitude', 0.0)
        )

        evaluate_kwargs = {
            'iterations': input_data.get('iterations', None)
        }

        y, iterations, times = self.so_task(cso, **evaluate_kwargs)

        # save to csv y and iterations
        if self.save_csv_details:
            write_csv(
                f'{user_input}.csv',
                (f'{user_input}_solution',
                 f'{user_input}_iterations',
                 f'{user_input}_runtime'),
                zip(y, iterations, times)
            )
        # take only actual repeats (from the last input)
        cur_y_matrix = self.y_matrix[-self.repeats:]
        # save all y-s in every repeat
        if self.save_csv_y_matrix:
            write_csv(
                f'{user_input}_y_matrix.csv',
                range(1, self.repeats + 1),
                # needs to rotate matrix
                zip_longest(*cur_y_matrix[::-1])
            )

    def glpso_task(self, user_input, input_data):
        # init classes
        opt_function = OptimizationFunction(input_data['function'])
        glpso = GLPSO(
            input_data['population'],
            input_data['dimension'],
            opt_function,
            input_data['pm'],
            levy=input_data.get('levy', False),
            **self.w_parameters[input_data['w_set']]
        )

        evaluate_kwargs = {
            'iterations': input_data.get('iterations', None)
        }

        y, iterations, times = self.so_task(glpso, **evaluate_kwargs)

        # save to csv y and iterations
        if self.save_csv_details:
            write_csv(
                f'{user_input}.csv',
                (f'{user_input}_solution',
                 f'{user_input}_iterations',
                 f'{user_input}_runtime'),
                zip(y, iterations, times)
            )
        # take only actual repeats (from the last input)
        cur_y_matrix = self.y_matrix[-self.repeats:]
        # save all y-s in every repeat
        if self.save_csv_y_matrix:
            write_csv(
                f'{user_input}_y_matrix.csv',
                range(1, self.repeats + 1),
                # needs to rotate matrix
                zip_longest(*cur_y_matrix[::-1])
            )

    def ba_task(self, user_input, input_data):
        # init classes
        opt_function = OptimizationFunction(input_data['function'])
        pso = PSO(
            input_data['population'],
            input_data['dimension'],
            opt_function,
            levy=input_data.get('levy', False),
            **input_data.get('parameters', None)
        )

        evaluate_kwargs = {
            'iterations': input_data.get('iterations', None)
        }

        y, iterations, times = self.so_task(pso, **evaluate_kwargs)

        # save to csv y and iterations
        if self.save_csv_details:
            write_csv(
                f'{user_input}.csv',
                (f'{user_input}_solution',
                 f'{user_input}_iterations',
                 f'{user_input}_runtime'),
                zip(y, iterations, times)
            )
        # take only actual repeats (from the last input)
        cur_y_matrix = self.y_matrix[-self.repeats:]
        # save all y-s in every repeat
        if self.save_csv_y_matrix:
            write_csv(
                f'{user_input}_y_matrix.csv',
                range(1, self.repeats + 1),
                # needs to rotate matrix
                zip_longest(*cur_y_matrix[::-1])
            )
