from ortools.algorithms.python import knapsack_solver
import time
import numpy as np

# running:
#   git clone https://github.com/likr/kplib
#   python main.py

def Read_Data(path_data):
    arr = {}
    arr['weight'] = []
    arr['values'] = []
    with open(path_data + '/R01000/s000.kp', 'rb') as file:
        data = file.read()

    f = data.split(b'\n')
    decoded_lines = [line.decode('utf-8') for line in f]  # Assuming UTF-8 encoding; adjust if necessary

    n = 0
    c = 0
    for line in range(0, len(decoded_lines)):
        decoded_lines[line] = decoded_lines[line].split(' ')
        if decoded_lines[line][0] == '':
            continue
        decoded_lines[line] = list(map(int, decoded_lines[line]))
        if len(decoded_lines[line]) == 0:
            continue
        elif len(decoded_lines[line]) == 1:
            if n == 0:
                n = decoded_lines[line][0]
            else:
                c = decoded_lines[line][0]
        else:
            arr['weight'].append(decoded_lines[line][1])
            arr['values'].append(decoded_lines[line][0])
    values = arr['values']
    weights = arr['weight']
    capacities = c

    return values, [weights], [capacities]

def main():
    test_cases = ['n00050', 'n00100', 'n00200', 'n00500', 'n01000', 'n02000']

    # Create the solver.
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )
    solver.set_time_limit(180)
    for test_case in test_cases:

        # Create the data
        values, weights, capacities = Read_Data(path + test_case)

        # Call the solver
        t1 = time.time()
        solver.init(values, weights, capacities)
        computed_value = solver.solve()
        t2 = time.time()

        packed_items = []
        packed_weights = []
        total_weight = 0

        for i in range(len(values)):
            if solver.best_solution_contains(i):
                packed_items.append(i)
                packed_weights.append(weights[0][i])
                total_weight += weights[0][i]

        print(f'Is optimal: {solver.is_solution_optimal()}\n')


if __name__ == "__main__":
    path = "./kplib/02StronglyCorrelated/"
    main()