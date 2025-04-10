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
        Initializes a new instance of the TransportationProblemSolver class.

        :param sets_home: Dictionary with all home slots by team.
        :param sets_forbidden: Dictionary with all forbidden slots by team.
        :param m: Minimum number of time slots between 2 games with same pair of teams.
            --> e.g., one game at slot t and the other game at slot t + m is allowed
                but at slot t + m - 1 is disallowed
        :param P: Cost from dummy supply node q to non-dummy demand node.
        :param R_max: Minimum required time slots for 2 games of same team.
            --> e.g., a single team can play a game at slot t and one as from
                slot t + R_max - 1 (as 'R_max' slots range from t to t + R_max - 1)
        :param penalties: Dictionary as {n_days: penalty} where n_days = rest days + 1.
            --> e.g., respective penalty is assigned if already 1 game
                between slot t - n_days and t + n_days excl. t
        """
        # set penalties default
        if penalties is None:
            penalties = {1: 10, 2: 3, 3: 1}

        self.sets_home = sets_home
        self.sets_forbidden = sets_forbidden
        self.m = m
        self.P = P
        self.R_max = max(2, R_max)  # must be at least 2
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

    # fmt: off
    def create_cost_matrix(
        self,
        X: np.ndarray,
        team_idx: int,
        set_home: dict,
        opponents: list,
    ) -> np.ndarray:
        """Creates costs in adjacency matrix based on current schedule & constraints."""
        am_cost = np.zeros((len(set_home), len(opponents)))

        games_team = np.concatenate((X[team_idx, :], X[:, team_idx]))

        home_dates = np.array(list(set_home)) # C2 - home date availability
        home_dates_r = home_dates.reshape(-1, 1)  # convert into column vector Nx1

        # NOTE: Gets broadcasted into 1xK - Nx1 => NxK matrix such that
        # a single row (k = 1, ..., K) has all current game slots minus a specific available home slot (n = 1, ..., N)
        games_team_d = np.abs(games_team.reshape(1, -1) - home_dates_r) + 1  # team game distances for all home dates

        for j, oppo_idx in enumerate(opponents):
            games_oppo = np.concatenate((X[oppo_idx, :], X[:, oppo_idx]))

            # C3 - forbidden game set
            forbidden_mask = np.array([h in self.sets_forbidden[oppo_idx] for h in home_dates])

            # C4 - team already plays game
            team_plays_mask = np.isin(home_dates, games_team)

            # C4 - opponent already plays game
            oppo_plays_mask = np.isin(home_dates, games_oppo)

            # C5 - max. 2 games for 'R_max' slots (e.g., R_max=7 allows at most 2 games between dates 01 -> 07 / t -> t + 6)
            # NOTE: For [t, h = t + x] we cannot have nbr. of slots x + 1 < R_max
            games_oppo_d = np.abs(games_oppo.reshape(1, -1) - home_dates_r) + 1  # opponent game distances for all home dates
            games_in_r_max_team = (games_team_d < self.R_max).sum(axis=1) > 0  # NOTE: Sums across columns to get info per available home slot
            games_in_r_max_oppo = (games_oppo_d < self.R_max).sum(axis=1) > 0
            r_max_mask = games_in_r_max_team | games_in_r_max_oppo

            # C6 - game i-j is within m days of game j-i
            reciprocal_game_mask = np.abs(home_dates - X[oppo_idx, team_idx]) < self.m

            # set disallowed cost for all disallowed slots at once
            disallowed = (
                forbidden_mask |
                team_plays_mask |
                oppo_plays_mask |
                reciprocal_game_mask |
                r_max_mask
            )
            am_cost[disallowed, j] = DISALLOWED_NBR

            # process penalty calculations for allowed slots
            allowed_indices = np.where(~disallowed)[0]
            if len(allowed_indices) > 0:
                for i in allowed_indices:
                    h = home_dates[i]

                    penalties = 0
                    for delta_games in [
                        games_team - h,  # forward-looking for team
                        h - games_team,  # backward-looking for team
                        games_oppo - h,  # forward-looking for opponent
                        h - games_oppo,  # backward-looking for opponent
                    ]:
                        # only positive values respect direction, then take minimum
                        delta_games_pos = delta_games[delta_games > 0]
                        if len(delta_games_pos) != 0:
                            penalties += self.penalties.get(np.nanmin(delta_games_pos), 0)

                    am_cost[i, j] = penalties

        return am_cost
    # fmt: on

    def get_total_cost(self, indexes: list[tuple[int, int]], am: list) -> float:
        """Returns total cost in adjacency matrix from optimal indexes."""
        return sum([am[row][column] for row, column in indexes])
