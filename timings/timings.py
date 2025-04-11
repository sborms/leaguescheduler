import time
from contextlib import contextmanager

import pandas as pd

from leaguescheduler import InputParser, LeagueScheduler, SchedulerParams

N_TEAMS_LIST = [13, 4]
N_ITERATIONS_LIST = [10, 100, 1000, 10000]

input_file = "example_input.xlsx"
output_file = "timings/timings_base.txt"
sheet_name = "LEAGUE A"
seed = 505


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


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


results = {}
for n_teams in N_TEAMS_LIST:
    lst_elapsed = []
    for n_iterations in N_ITERATIONS_LIST:
        with timer() as elapsed:
            optimize(n_teams, n_iterations)
        elapsed_seconds = elapsed()
        lst_elapsed.append(elapsed_seconds)
        print(f"{n_teams} teams | {n_iterations} its. => time: {elapsed_seconds:.2f}s")
    results[n_teams] = lst_elapsed

df = pd.DataFrame(results, index=N_ITERATIONS_LIST).round(2)

print("\nTimings:")
print(df.to_string(index=True))

with open(output_file, "w") as f:
    f.write(df.to_string(index=True))
