import time
from contextlib import contextmanager

import pandas as pd

from leaguescheduler import InputParser, LeagueScheduler

N_ITERATIONS_LIST = [10, 50, 100, 500, 1000, 5000]

input_file = "example_input.xlsx"
output_file = "experiments/timings_base.txt"
seed = 505


def optimize(n_iterations):
    input = InputParser(input_file)

    for sheet_name in input.sheet_names:
        input.read(sheet_name=sheet_name)
        input.parse()

        scheduler = LeagueScheduler(
            input=input,
            n_iterations=n_iterations,
            penalties=input.penalties,
        )
        scheduler.construction_phase()
        scheduler.tabu_phase()


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


results = []
for n_iterations in N_ITERATIONS_LIST:
    with timer() as elapsed:
        optimize(n_iterations)
    elapsed_seconds = elapsed()
    results.append(elapsed_seconds)
    print(f"{n_iterations} iterations => elapsed: {elapsed_seconds:.2f} seconds")

df = pd.DataFrame({"elapsed_seconds": results}, index=N_ITERATIONS_LIST)
df["elapsed_seconds"] = df["elapsed_seconds"].round(2)

print("\nTimings:")
print(df.to_string(index=True))

with open(output_file, "w") as f:
    f.write(df.to_string(index=True))
