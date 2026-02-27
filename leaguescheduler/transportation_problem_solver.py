import fasttps
import numpy as np

from .params import SchedulerParams

DEFAULTS = SchedulerParams()


class TransportationProblemSolver:
    """Helper class to solve transportation problem."""

    def __init__(
        self,
        sets_home: dict,
        sets_forbidden: dict,
        m: int = DEFAULTS.m,
        p: int = DEFAULTS.p,
        r_max: int = DEFAULTS.r_max,
        penalties: dict[int, int] = DEFAULTS.penalties,
    ) -> None:
        """
        Initializes a new instance of the TransportationProblemSolver class.

        :param sets_home: Dictionary with all home slots by team.
        :param sets_forbidden: Dictionary with all forbidden slots by team.
        :param m: See SchedulerParams for parameter details.
        :param p: See SchedulerParams for parameter details.
        :param r_max: See SchedulerParams for parameter details.
        :param penalties: See SchedulerParams for parameter details.
        """
        self.sets_home = sets_home
        self.sets_forbidden = sets_forbidden
        self.m = m
        self.p = p
        self.r_max = max(2, r_max)  # must be at least 2
        self.penalties = penalties

        # Rust-backed solver (stores precomputed state for fast repeated calls)
        self._rust = fasttps.FastTPS(
            sets_home=sets_home,
            sets_forbidden=sets_forbidden,
            m=m,
            p=p,
            r_max=r_max,
            penalties=penalties,
        )

    def solve(
        self,
        X: np.ndarray,
        team_idx: int,
    ) -> tuple[list, int]:
        """
        Solves transportation problem for given home team (= row) and set of home slots.
        Returns updated X alongside cost from adjacency matrix and picked indexes.
        """
        total_cost = self._rust.solve(X, team_idx)
        return X, total_cost

    def create_cost_matrix(
        self,
        X: np.ndarray,
        team_idx: int,
        set_home: dict,
        opponents: list,
    ) -> np.ndarray:
        """Creates costs in adjacency matrix based on current schedule & constraints."""
        return self._rust.create_cost_matrix(
            X, team_idx, list(set_home), list(opponents)
        )

    def get_total_cost(
        self,
        indexes: list[tuple[int, int]],
        am: list,
    ) -> float:
        """Returns total cost in adjacency matrix from optimal indexes."""
        return sum([am[row][column] for row, column in indexes])
