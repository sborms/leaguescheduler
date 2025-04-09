import numpy as np
from munkres import DISALLOWED as D
from munkres import Munkres

from .constants import DISALLOWED_NBR


class TransportationProblemSolver:
    """Helper class to solve transportation problem."""

    def __init__(
        self,
        sets_home: dict,
        sets_forbidden: dict,
        m: int = 14,
        P: int = 1000,
        R_max: int = 4,
        penalties: dict = None,
    ) -> None:
        """
        Initializes a new instance of the Transportation class.

        :param sets_home: Dictionary with all home slots by team.
        :param sets_forbidden: Dictionary with all forbidden slots by team.
        :param m: Minimum number of time slots between 2 games with same pair of teams.
        :param P: Cost from dummy supply node q to non-dummy demand node.
        :param R_max: Minimum required time slots for 2 games of same team.
        :param penalties: Dictionary as {n_days: penalty} where n_days = rest days + 1
            --> e.g., respective penalty is assigned if already 1 game
                between slot t - n_days and t + n_days excl. t.
        """
        # set penalties default
        if penalties is None:
            penalties = {1: 10, 2: 3, 3: 1}

        self.sets_home = sets_home
        self.sets_forbidden = sets_forbidden
        self.m = m
        self.P = P
        self.R_max = R_max
        self.penalties = penalties

    def solve(self, X: np.ndarray, team_idx: int) -> tuple[list, int]:
        """
        Solves transportation problem for given home team (= row) and set of home slots.
        Returns updated X along with cost from adjacency matrix and picked indexes.
        """
        set_home = self.sets_home[team_idx]
        opponents = [t for t in range(X.shape[0]) if t != team_idx]

        n_set_home = len(set_home)
        n_opponents = len(opponents)
        dim = n_set_home + n_opponents

        # construct components of adjacency matrix (= am)
        am_cost = self.create_cost_matrix(X, team_idx, set_home, opponents)
        am_bott = np.full((n_opponents, n_opponents), self.P)
        am_righ = np.zeros((dim, dim - n_opponents))

        # construct full original adjacency matrix
        am_np = np.concatenate(
            (
                np.concatenate((am_cost, am_bott), axis=0),
                am_righ,
            ),
            axis=1,
        )

        # convert to list and add DISALLOWED constant
        am = [[D if v == DISALLOWED_NBR else v for v in r] for r in am_np.tolist()]

        # run Hungarian algorithm to solve transportation problem
        m = Munkres()
        indexes = m.compute(am)
        total_cost = self.get_total_cost(indexes, am)

        # process optimal indexes (np.nan means not yet scheduled)
        indexes_inv = sorted(
            (opponents[v], set_home[k] if k < len(set_home) else np.nan)
            for k, v in indexes
            if v < n_opponents
        )
        pick = [v for _, v in indexes_inv]

        # assign selection to X
        X[team_idx, opponents] = pick

        return X, total_cost

    def create_cost_matrix(
        self,
        X: np.ndarray,
        team_idx: int,
        set_home: dict,
        opponents: list,
    ) -> np.ndarray:
        """Creates costs in adjacency matrix based on current schedule & constraints."""
        am_cost = np.zeros((len(set_home), len(opponents)))
        for j, oppo_idx in enumerate(opponents):  # C1
            games_team = np.concatenate((X[team_idx, :], X[:, team_idx]))
            games_oppo = np.concatenate((X[oppo_idx, :], X[:, oppo_idx]))

            # fmt: off
            for i, h in enumerate(set_home):  # C2
                games_team_w = abs(games_team - h)
                games_oppo_w = abs(games_oppo - h)

                # forbidden game set
                if h in self.sets_forbidden[oppo_idx]:  # C3
                    am_cost[i, j] = DISALLOWED_NBR
                # team already plays away game
                elif h in games_team:  # C4
                    am_cost[i, j] = DISALLOWED_NBR
                # opponent already plays home/away game
                elif h in games_oppo:  # C4
                    am_cost[i, j] = DISALLOWED_NBR
                # already 2 (aka >1) games within 'R_max' slots (e.g. 7 allows >1 game between dates 01 -> 07)
                elif sum(games_team_w < self.R_max) > 1 or sum(games_oppo_w < self.R_max) > 1:  # C5
                    am_cost[i, j] = DISALLOWED_NBR
                # game i-j is within m days of game j-i
                elif abs(h - X[oppo_idx, team_idx]) < self.m:  # C6
                    am_cost[i, j] = DISALLOWED_NBR

                if am_cost[i, j] == DISALLOWED_NBR:
                    continue

                # add penalties for closest game in past and future for both teams
                list_delta_games = [
                    games_team - h,  # forward-looking for team
                    h - games_team,  # backward-looking for team
                    games_oppo - h,  # forward-looking for opponent
                    h - games_oppo,  # backward-looking for opponent
                ]
                for delta_games in list_delta_games:
                    # only positive values respect direction, then take minimum
                    delta_games_pos = delta_games[delta_games > 0]
                    if len(delta_games_pos) != 0:
                        am_cost[i, j] += self.penalties.get(np.nanmin(delta_games_pos), 0)
            # fmt: on

        return am_cost

    def get_total_cost(self, indexes: list[tuple[int, int]], am: list) -> float:
        """Returns total cost in adjacency matrix from optimal indexes."""
        return sum([am[row][column] for row, column in indexes])
