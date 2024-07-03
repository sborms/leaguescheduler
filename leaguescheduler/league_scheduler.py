import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .constants import LARGE_NBR, OUTPUT_COLS
from .input_parser import InputParser
from .transportation_problem_solver import TransportationProblemSolver as TPS

# NOTE: Below minor features of the algorithm are NOT YET IMPLEMENTED
# Minimal cost change in tabu phase
# Dealing with teams providing more than one home slot within 'R_max' slots

# NOTE: These are common reasons why a game remains unscheduled
# No (or too little) home availabilities
# No away availabilities on home days
# Some teams have home games on same day

# TODO: What is impact of nbr. of iterations on optimality in terms of min. and max. rest days?
# TODO: Give final and all needed output(s)


class Perturbation:
    """Helper class to modify current schedule to avoid local optima."""

    def __init__(self, alpha: float = 0.50, beta: float = 0.01) -> None:
        """
        Initializes a new instance of the Perturbation class.

        :param alpha: Picks perturbation operator 1 with probability alpha.
        :param beta: Probability of removing a game in operator 1.
        """
        self.alpha = alpha
        self.beta = beta

    def perturbate(self, X: np.ndarray) -> None:
        """Perturbates the given matrix in-place with the first or second operator."""
        if np.random.rand() < self.alpha:
            self.perturbate1(X)
        else:
            self.perturbate2(X)

    def perturbate1(self, X: np.ndarray) -> None:
        """
        First perturbation operator
        --> Randomly determines for each game in the schedule independently if the
            game is to be removed with probability self.beta.

        Picked with probability self.alpha.
        """
        X[np.random.rand(*X.shape) < self.beta] = np.nan
        np.fill_diagonal(X, LARGE_NBR)

    def perturbate2(self, X: np.ndarray) -> None:
        """
        Second perturbation operator
        --> Chooses a team with a uniform probability, removes all the games of this
            team, and solves the transportation problem for this team.

        Picked with probability 1 - self.alpha.
        """
        idx = np.random.choice(range(X.shape[0]))

        X[idx, :] = np.nan
        X[:, idx] = np.nan
        X[idx, idx] = LARGE_NBR


class LeagueScheduler:
    """
    Generates an optimal schedule for a time-relaxed double round-robin (2RR) league
    accounting for following constraints:
        (C1) Each team plays a home game against each other team at most once.
        (C2) Each home team's availability set (H) is respected.
        (C3) Each away team's unavailability set (A) is respected.
        (C4) Each team plays at most one game per time slot.
        (C5) Each team plays at most 2 games in a period of 'R_max' time slots.
        (C6) There are minimum 'm' time slots between two games with the same teams (pairs).

    The implementation very closely follows the tabu search based algorithm from:
    Van Bulck, D., Goossens, D. R., & Spieksma, F. C. R. (2019).
    Scheduling a non-professional indoor football league: a tabu search based approach.
    Annals of Operations Research, 275(2), 715-730.
    https://doi.org/10.1007/s10479-018-3013-x
    """

    def __init__(
        self,
        input: InputParser,
        tabu_length: int = 4,
        perturbation_length: int = 50,
        n_iterations: int = 1500,
        m: int = 14,
        P: int = 5000,
        R_max: int = 4,
        penalties: dict = {1: 10, 2: 3, 3: 1},
        alpha: float = 0.50,
        beta: float = 0.01,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """
        Initializes a new instance of the LeagueScheduler class.

        :param input: InputParser object containing all relevant data.
        :param tabu_length: Number of iterations during which a team cannot be selected.
        :param perturbation_length: Check perturbation need every this many iterations.
        :param n_iterations: Number of tabu phase iterations.
        :param m: Minimum number of time slots between 2 games with same pair of teams.
        :param P: Cost from dummy supply node q to non-dummy demand node.
        :param R_max: Ideal minimum time slots for 2 games of same team.
        :param penalties: Dictionary as {n_days: penalty} where n_days ~ rest days + 1
            --> e.g., respective penalty is assigned if already 1 game
                between slot t - n_days and t + n_days excl. t.
        :param alpha: Picks perturbation operator 1 with probability alpha.
        :param beta: Probability of removing a game in operator 1.
        :param logger: (optional) Logger instance for logging purposes.
        """
        # assign input data to carry along
        self.input = input
        self.tabu_length = tabu_length
        self.perturbation_length = perturbation_length
        self.n_iterations = n_iterations
        self.logger = logger

        # initialize target matrix with teams & slots
        X = np.eye(len(input.sets["teams"])) * LARGE_NBR  # diagonal is to be ignored
        self.X = np.where(X == 0, np.nan, X)

        # initialize transportation object
        self.tps = TPS(
            sets_forbidden=input.sets["forbidden"],
            sets_home=input.sets["home"],
            m=m,
            P=P,
            R_max=R_max,
            penalties=penalties,
        )

        # initialize perturbation object
        self.perturbation = Perturbation(alpha=alpha, beta=beta)

        # initialize output columns
        self.output_cols = OUTPUT_COLS

    def construction_phase(self) -> None:
        """Generates initial (possibly incomplete) schedule and assigns it to self.X."""
        X = self.X
        n_teams = len(self.input.sets["teams"])

        # initialize list with costs per home team
        self.list_home_costs = [None] * n_teams

        # method 1
        # repeatedly select team with smallest number of available home slots
        X1 = X.copy()
        list_home_costs1 = self.list_home_costs.copy()
        d_spots1 = self._update_dict_available_spots(method=1)  # initialize dict

        for _ in range(n_teams):
            team_idx = list(d_spots1)[0]  # pick team

            # solve transportation problem for home team in current schedule X1
            X1, total_cost = self.tps.solve(X1, team_idx)
            list_home_costs1[team_idx] = total_cost

            # update available spots
            d_spots1 = self._update_dict_available_spots(1, X1, d_spots1, team_idx)

        cost1 = sum(list_home_costs1)
        self.logger.info(f"Initialized schedule using method 1 with cost {cost1}")

        # method 2
        # repeatedly select team with smallest number of possible games
        X2 = X.copy()
        list_home_costs2 = self.list_home_costs.copy()
        d_spots2 = self._update_dict_available_spots(method=2, X=X2)  # initialize dict

        for _ in range(n_teams):
            team_idx = list(d_spots2)[0]  # pick team

            # solve transportation problem for home team in current schedule X2
            X2, total_cost = self.tps.solve(X2, team_idx)
            list_home_costs2[team_idx] = total_cost

            # update available spots
            d_spots2 = self._update_dict_available_spots(2, X2, d_spots2, team_idx)

        cost2 = sum(list_home_costs2)
        self.logger.info(f"Initialized schedule using method 2 with cost {cost2}")

        # pick best method to set schedule after construction phase
        if cost1 < cost2:
            self.logger.info(f"Initialization method 1 is best")
            self.X, self.list_home_costs = X1, list_home_costs1
        else:
            self.logger.info(f"Initialization method 2 is best")
            self.X, self.list_home_costs = X2, list_home_costs2

    def tabu_phase(
        self, progress_bar: st.delta_generator.DeltaGenerator = None
    ) -> None:
        """
        Solves transportation problem to (re)schedule all home games of
        a non-tabu team (= not recently chosen), for a certain number of
        iterations or until the full cost reached zero. Every new optimal
        schedule is added to self.X.

        :param progress_bar: A progress bar object, e.g., streamlit.progress(0.0).
        """
        X = self.X.copy()  # get current schedule

        list_home_costs = self.list_home_costs.copy()

        # initialize list with full costs at end of every iteration
        self.list_full_costs = [sum(list_home_costs)]
        full_cost_min = self.list_full_costs[-1]
        self.logger.info(f"Tabu phase starts with cost {full_cost_min}")

        list_tabu = []
        list_nontabu = list(self.input.sets["teams"].keys())

        it = 0
        while it < self.n_iterations and self.list_full_costs[-1] > 0:
            it += 1
            if progress_bar is not None:
                progress_bar.progress(it / self.n_iterations)

            # check if current schedule needs to be perturbated
            if it % self.perturbation_length == 0:
                # no better solution found
                if (
                    self.list_full_costs[-1]
                    >= self.list_full_costs[-self.perturbation_length]
                ):
                    n_unsched_pre = np.sum(np.isnan(X), axis=1)
                    self.logger.info(f"Perturbating schedule at iteration {it}")
                    self.perturbation.perturbate(X)

                    # adjust costs based on dropped games from perturbation
                    n_unsched_pos = np.sum(np.isnan(X), axis=1)
                    list_home_costs += (n_unsched_pos - n_unsched_pre) * self.tps.P

            # recover team that has been tabu_length iterations in tabu list
            if it > self.tabu_length:
                team_nontabu = list_tabu.pop(0)
                list_nontabu.append(team_nontabu)

            # randomly choose non-tabu team
            team_idx = np.random.choice(list_nontabu)
            list_tabu.append(team_idx)
            list_nontabu.remove(team_idx)

            # reschedule home games of picked team
            X[team_idx, :] = np.nan
            X[team_idx, team_idx] = LARGE_NBR

            X, total_cost = self.tps.solve(X, team_idx)  # solver

            # update costs
            list_home_costs[team_idx] = total_cost
            full_cost = sum(list_home_costs)
            self.list_full_costs.append(full_cost)

            # check quality
            if full_cost < full_cost_min:  # new best
                full_cost_min = full_cost
                self.logger.info(
                    f"New best at iteration {it} with cost {full_cost_min}"
                )
                self.X = X.copy()  # add new optimal schedule

        # set progress bar to 100% (needed in case of early termination)
        if progress_bar is not None:
            progress_bar.progress(1.0)

    def plot_minimum_costs(self, path: str = None) -> None:
        """Plots evolution of running minimum cost during tabu phase."""
        if not hasattr(self, "list_full_costs"):
            self.logger.warning(
                "No costs available for plotting, run self.tabu_phase() first"
            )
            return

        list_running_minimum_cost = [
            min(self.list_full_costs[: i + 1]) for i in range(len(self.list_full_costs))
        ]

        # create plot
        plt.figure(figsize=(10, 5))
        plt.plot(list_running_minimum_cost)
        plt.title("Evolution minimum cost")
        plt.xlabel("Iteration")
        plt.tight_layout()

        # show or save plot
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    def create_calendar(self) -> pd.DataFrame:
        """Creates a calendar DataFrame from the optimal schedule."""
        X = self.X
        core = self.input.core
        teams = self.input.sets["teams"]
        locations = self.input.locations
        set_slots = self.input.sets["slots"]

        list_team, list_oppo, list_location, list_date, list_hour = [], [], [], [], []

        for i in range(X.shape[0]):
            team = teams[i]
            for j in range(X.shape[1]):
                if i == j:
                    continue
                oppo = teams[j]

                list_team.append(team)
                list_oppo.append(oppo)
                list_location.append(locations[team])

                slot = X[i, j]
                if not pd.isna(slot):
                    list_date.append(set_slots[slot])
                    list_hour.append(core[team].loc[slot])
                else:
                    list_date.append(np.nan)
                    list_hour.append(np.nan)

        df = pd.DataFrame(
            {
                self.output_cols[0]: list_date,
                self.output_cols[1]: list_hour,
                self.output_cols[2]: list_location,
                self.output_cols[3]: list_team,
                self.output_cols[4]: list_oppo,
            }
        )

        df[self.output_cols[0]] = pd.to_datetime(df[self.output_cols[0]])
        df = df.sort_values(self.output_cols[0])

        return df

    def store_calendar(self, df: pd.DataFrame, file: str) -> None:
        """Stores generated calendar as an Excel file."""
        df_out = df.copy()
        df_out[self.output_cols[0]] = df_out[self.output_cols[0]].dt.strftime(
            "%Y-%m-%d"
        )

        df_out[self.output_cols].to_excel(file, index=False)

    def validate(self, df: pd.DataFrame) -> dict:
        """Gathers a dictionary with validation data on the generated schedule."""
        d_val = {}

        # grab some general statistics first
        d_val["teams"] = len(self.input.sets["teams"])
        d_val["games"] = len(df)
        d_val["unscheduled"] = sum(df[self.output_cols[0]].isna())
        d_val["cost"] = min(self.list_full_costs)

        # overview of total number of home slots less than needed (per-team basis)
        n_home_slots_short = sum(
            [
                max((len(self.input.sets["teams"]) - 1) - len(v), 0)
                for _, v in self.input.sets["home"].items()
            ]
        )
        d_val["missing_home_slots"] = n_home_slots_short

        # overview of number of games between two teams
        df["pairs"] = df.apply(
            lambda row: tuple(
                sorted([row[self.output_cols[3]], row[self.output_cols[4]]])
            ),
            axis=1,
        )

        d_val["pairs"] = df["pairs"].value_counts()

        # overview of days between games per pair of teams
        df["days_diff"] = df.groupby("pairs")[self.output_cols[0]].diff().dt.days
        df_days_diff_pairs = (
            df[["pairs", "days_diff"]]
            .dropna()
            .sort_values("days_diff")
            .reset_index(drop=True)
        )

        d_val["min_gap_pairs"] = df_days_diff_pairs["days_diff"].min()
        d_val["max_gap_pairs"] = df_days_diff_pairs["days_diff"].max()

        # overview of rest days in matrix form
        d_val["df_rest_days"] = self.make_df_rest_days(df)

        # overview of unused home slots
        d_val["df_unused_home_slots"] = self.make_df_unused_home_slots()

        return d_val

    def make_df_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forms 'matrix' of teams vs. number of rest days for given schedule."""
        col_date = self.output_cols[0]
        col_home = self.output_cols[3]
        col_away = self.output_cols[4]
        col_team = "Team"

        df[col_date] = pd.to_datetime(df[col_date])

        df_teams = pd.concat(
            [
                df[[col_date, col_home]].rename(columns={col_home: col_team}),
                df[[col_date, col_away]].rename(columns={col_away: col_team}),
            ]
        ).sort_values([col_team, col_date])
        df_teams["n_rest_days"] = (
            df_teams.groupby(col_team)[col_date].diff().dt.days - 1
        )

        mat = (
            df_teams.groupby(col_team)["n_rest_days"].value_counts().unstack().fillna(0)
        )
        mat = pd.concat(
            [mat, pd.DataFrame(mat.sum(axis=0), columns=["TOTAL"]).transpose()], axis=0
        )

        return mat

    def make_df_unused_home_slots(self) -> pd.DataFrame:
        """Forms DataFrame with teams and their unused home slots."""
        list_unused = []
        for team_idx, team_name in self.input.sets["teams"].items():
            unused_home_slots = sorted(
                [
                    self.input.sets["slots"][s]
                    for s in set(self.input.sets["home"][team_idx]).difference(
                        set(self.X[team_idx, :])
                    )
                ]
            )

            df_unused = pd.DataFrame({"unused": unused_home_slots})
            if len(df_unused) == 0:
                continue
            df_unused["team"] = team_name
            list_unused.append(df_unused)

        df_unused_all = pd.concat(list_unused)[["team", "unused"]].reset_index(
            drop=True
        )

        return df_unused_all

    def _update_dict_available_spots(
        self,
        method: int,
        X: np.ndarray = None,
        d_spots: dict = None,
        team_idx_last: int = None,
    ) -> dict:
        """Updates available home/game spots for each team during construction phase."""
        sets_home = self.input.sets["home"]

        if team_idx_last is not None:
            d_spots.pop(team_idx_last)  # drop last processed team

        if method == 1:
            if d_spots is None:
                # initialize spots from available home time slots
                d_spots = {key: len(sets_home[key]) for key in self.input.sets["teams"]}
            else:
                # subtract current scheduled away games
                d_spots = {
                    key: v - np.sum(np.isin(sets_home[2], X[:, 2]))
                    for key, v in d_spots.items()
                }
        elif method == 2:
            if d_spots is None:
                d_spots = {key: None for key in self.input.sets["teams"]}
            for team_idx in d_spots:
                set_home = self.tps.sets_home[team_idx]
                opponents = [t for t in range(X.shape[0]) if t != team_idx]

                m = self.tps.create_cost_matrix(X, team_idx, set_home, opponents)

                # count number of home slots possible for each opponent
                home_option_score = np.sum(m == 0, axis=0).min()
                d_spots[team_idx] = home_option_score

        # sort by number of spots (low to high)
        d_spots = dict(sorted(d_spots.items(), key=lambda x: x[1]))

        return d_spots
