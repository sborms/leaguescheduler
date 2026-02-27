import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from tests.utils import optimize

OVERWRITE_ORACLE = False

N_TEAMS_LIST = [13]
N_ITERATIONS_LIST = [10, 100, 1000, 10000, 100000, 1000000]


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


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

    results_time[n_teams] = list_elapsed
    results_X[n_teams] = list_X

df = pd.DataFrame(results_time, index=N_ITERATIONS_LIST).round(2)
df_str_repr = df.to_string(index=True)

print("\nTimings:")
print(df_str_repr)

# save results_X as oracle for future verification
if OVERWRITE_ORACLE:
    np.savez(
        "tests/oracle.npz",
        **{
            f"teams{n_teams}_iter{it}": X
            for n_teams, xs in results_X.items()
            for it, X in zip(N_ITERATIONS_LIST, xs, strict=False)
        },
    )
