import json

from task_manager import TaskManager


def check_input(predicate, msg, error_string='Input does not exist'):
    while True:
        result = input(msg).strip()
        if predicate(result):
            return result
        print(error_string)


if __name__ == '__main__':
    with open('files/input.json') as json_file:
        data = json.load(json_file)
    assert isinstance(data, dict)

    # auto pass inputs
    if data['settings']['auto']:
        user_inputs = data['settings']['inputs']
        inputs_data = [data[auto_input] for auto_input in user_inputs]
    # pass input by choosing one from list
    else:
        [print(f'{k} : {v}') for k, v in data.items()
         if k.startswith(('pso_', 'lcso_', 'cso_', 'glpso', 'ba'))]
        user_inputs = check_input(lambda x: x in data, 'Choose your input: ')
        inputs_data = [data[user_inputs]]
        user_inputs = [user_inputs]
    # user inputs e.g. ['pso_f10', 'pso_a', 'pso_b']
    # inputs data grouped for each user input
    # e.g. [{'function': 'f1', 'population: '20' ...}, {...}, {...}]
    tm = TaskManager(data, user_inputs, inputs_data)
    tm.start_tasks()
