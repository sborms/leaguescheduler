import numpy as np
from munkres import DISALLOWED as D
from munkres import Munkres

from .constants import DISALLOWED_NBR, LARGE_NBR
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
        am_bott = np.full((n_opponents, n_opponents), self.p)
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

        games_team = self._get_team_array(X, team_idx)

        # C2 - home date availability
        home_dates = np.array(list(set_home))
        home_dates_r = home_dates.reshape(-1, 1)  # convert into column vector Nx1

        # NOTE: Gets broadcasted into 1xK - Nx1 => NxK matrix such that
        # a single row (k = 1, ..., K) has all current game slots minus a specific available home slot (n = 1, ..., N)
        games_team_d = np.abs(games_team.reshape(1, -1) - home_dates_r) + 1  # team game distances for all home dates

        # C4 - team already plays game
        team_plays_mask = np.isin(home_dates, games_team)

        # C5 - cf. below
        # NOTE: Sums across columns to get info per available home slot
        games_in_r_max_team = (games_team_d < self.r_max).sum(axis=1) > 0

        for j, oppo_idx in enumerate(opponents):
            games_oppo = self._get_team_array(X, oppo_idx)

            # C3 - forbidden game set
            forbidden_mask = np.array([h in self.sets_forbidden[oppo_idx] for h in home_dates], dtype=bool)

            # C4 - opponent already plays game
            oppo_plays_mask = np.isin(home_dates, games_oppo)

            # C5 - max. 2 games for 'r_max' slots (e.g., r_max=7 allows at most 2 games between dates 01 -> 07 / t -> t + 6)
            # NOTE: For [t, h = t + x] we cannot have nbr. of slots x + 1 < r_max
            games_oppo_d = np.abs(games_oppo.reshape(1, -1) - home_dates_r) + 1  # opponent game distances for all home dates
            games_in_r_max_oppo = (games_oppo_d < self.r_max).sum(axis=1) > 0
            r_max_mask = games_in_r_max_team | games_in_r_max_oppo

            # C6 - game i-j is within m days of game j-i
            reciprocal_game_mask = np.abs(home_dates - X[oppo_idx, team_idx]) < self.m

            # set disallowed cost for all disallowed slots at once
            # NOTE: This does not take into account the time slot differences between the hereafter
            # picked home slots (e.g. two home slots within r_max -> _get_feasible_home_slots)
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

                    forw_t = games_team - h  # forward-looking for team
                    back_t = h - games_team  # backward-looking for team
                    forw_o = games_oppo - h  # forward-looking for opponent
                    back_o = h - games_oppo  # backward-looking for opponent
                    deltas = [forw_t, back_t, forw_o, back_o]

                    penalties = 0
                    for delta in deltas:
                        # only positive values respect direction, then take minimum
                        delta_pos = delta[delta > 0]
                        if delta_pos.size > 0:
                            min_delta_pos = delta_pos.min()
                            penalties += self.penalties.get(min_delta_pos, 0)

                    am_cost[i, j] = penalties

        return am_cost
    # fmt: on

    def get_total_cost(self, indexes: list[tuple[int, int]], am: list) -> float:
        """Returns total cost in adjacency matrix from optimal indexes."""
        return sum([am[row][column] for row, column in indexes])

    def _get_team_array(self, X: np.ndarray, idx: int) -> np.ndarray:
        arr = np.concatenate((X[idx, :], X[:, idx]))
        return arr[arr != LARGE_NBR]
