from pathlib import Path

import numpy as np
import pytest
from utils import optimize

ORACLE_FILE = Path(__file__).parent / "oracle.npz"

N_TEAMS_LIST = [13, 4]
N_ITERATIONS_LIST = [10, 100, 1000, 10000]


@pytest.fixture(scope="module")
def oracle():
    return np.load(ORACLE_FILE)


@pytest.fixture(scope="module")
def results():
    np.random.seed(505)

    out = {}
    for n_teams in N_TEAMS_LIST:
        for n_iterations in N_ITERATIONS_LIST:
            key = f"teams{n_teams}_iter{n_iterations}"
            out[key] = optimize(n_teams, n_iterations)

    return out


@pytest.mark.parametrize("n_teams", N_TEAMS_LIST)
@pytest.mark.parametrize("n_iterations", N_ITERATIONS_LIST)
def test_oracle(oracle, results, n_teams, n_iterations):
    key = f"teams{n_teams}_iter{n_iterations}"
    oracle_X = oracle.get(key)

    assert oracle_X is not None, f"Oracle missing key {key}"
    assert np.array_equal(results[key], oracle_X), f"Mismatch for {key}"
