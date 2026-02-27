import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from leaguescheduler import InputParser, LeagueScheduler, SchedulerParams

N_TEAMS_LIST = [13, 4]

# N_ITERATIONS_LIST = [10, 100]  # oracle
N_ITERATIONS_LIST = [10, 100, 1000, 10000]  # oracle10k

input_file = "example_input.xlsx"
output_file = "timings/timings.txt"
oracle_file = "timings/oracle10k.npz"
sheet_name = "LEAGUE A"

np.random.seed(505)


def optimize(n_teams, n_iterations):
    input = InputParser(input_file)

    input.from_excel(sheet_name=sheet_name)
    input.parse()

    # extract the number of teams
    input.data = input.data.iloc[:, : n_teams + 1]

    scheduler = LeagueScheduler(
        input=input,
        params=SchedulerParams(n_iterations=n_iterations, penalties=input.penalties),
    )
    scheduler.construction_phase()
    scheduler.tabu_phase()

    return scheduler.X


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


# clear file
with open(output_file, "w") as f:
    pass

results_time, results_X = {}, {}
for n_teams in N_TEAMS_LIST:
    list_elapsed, list_X = [], []
    for n_iterations in N_ITERATIONS_LIST:
        with timer() as elapsed:
            X = optimize(n_teams, n_iterations)
        elapsed_seconds = elapsed()

        list_elapsed.append(elapsed_seconds)
        list_X.append(X)

        print(f"{n_teams} teams | {n_iterations} its. => time: {elapsed_seconds:.2f}s")

        with open(output_file, "a") as f:
            f.write(f"{n_teams} teams | {n_iterations} its. => solution:\n{X}\n\n")

    results_time[n_teams] = list_elapsed
    results_X[n_teams] = list_X

df = pd.DataFrame(results_time, index=N_ITERATIONS_LIST).round(2)
df_str_repr = df.to_string(index=True)

print("\nTimings:")
print(df_str_repr)

with open(output_file, "a") as f:
    f.write(f"Timings:\n{df_str_repr}\n")

# save results_X for later use
np.savez(
    "timings/solutions.npz",
    **{
        f"teams{n_teams}_iter{it}": X
        for n_teams, xs in results_X.items()
        for it, X in zip(N_ITERATIONS_LIST, xs, strict=False)
    },
)

# verify optimized results against oracle
oracle = np.load(oracle_file)

dict_checks = {}
for n_teams in N_TEAMS_LIST:
    for n_iterations in N_ITERATIONS_LIST:
        X = results_X[n_teams][N_ITERATIONS_LIST.index(n_iterations)]
        oracle_X = oracle.get(f"teams{n_teams}_iter{n_iterations}")

        if oracle_X is not None:
            is_equal = np.array_equal(X, oracle_X)
            dict_checks[f"teams{n_teams}_iter{n_iterations}"] = is_equal

print("\nVerification against oracle:")
print(f"{'Configuration':<20} | {'Check'}")
for key, is_equal in dict_checks.items():
    print(f"{key:<20} | {'OK' if is_equal else 'NOT OK'}")
