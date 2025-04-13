import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .constants import LARGE_NBR, OUTPUT_COLS
from .input_parser import InputParser
from .params import SchedulerParams
from .transportation_problem_solver import TransportationProblemSolver as TPS
from .utils import drop_nearby_points_from_array

# NOTE: These are common reasons why a game remains unscheduled
# No (or too little) home availabilities
# No away availabilities on home days
# Some teams have home games on same day


class Perturbation:
    """Helper class to modify current schedule to avoid local optima."""

    def __init__(self, alpha: float = 0.50, beta: float = 0.01) -> None:
        """
        Initializes a new instance of the Perturbation class.

        See SchedulerParams for parameter details.
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
    - (C1) Each team plays a home game against each other team at most once.
    - (C2) Each home team its availability set (H) is respected.
    - (C3) Each away team its unavailability set (A) is respected.
    - (C4) Each team plays at most one game per time slot.
    - (C5) Each team plays at most 2 games in a period of 'r_max' time slots.
    - (C6) There are minimum 'm' time slots between two games with the same teams (pairs).

    The implementation very closely follows the tabu search based algorithm from:
    > Van Bulck, D., Goossens, D. R., & Spieksma, F. C. R. (2019).
    _Scheduling a non-professional indoor football league: a tabu search based approach._
    Annals of Operations Research, 275(2), 715-730.
    https://doi.org/10.1007/s10479-018-3013-x

    A time slot uniquely maps to a weekday (with an associated playing hour). For
    instance, if slot t is a Monday, then slot t + 3 is a Thursday, with in total 4 slots
    considered. The number of rest days is 2 in this case (Tuesday and Wednesday).
    """

    def __init__(
        self,
        input: InputParser,
        params: SchedulerParams,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """
        Initializes a new instance of the LeagueScheduler class.

        See the [project README](https://github.com/sborms/leaguescheduler) for
        more information about usage.

        :param input: InputParser object containing all relevant data.
        :param params: See SchedulerParams for parameter details.
        :param logger: (optional) Logger instance for logging purposes.
        """
        assert input.parsed, "Input data not parsed yet!"

        # assign input data to carry along
        self.input = input
        self.tabu_length = params.tabu_length
        self.perturbation_length = params.perturbation_length
        self.n_iterations = params.n_iterations
        self.logger = logger

        # initialize target matrix with teams & slots
        X = np.eye(len(input.sets["teams"])) * LARGE_NBR  # diagonal is to be ignored
        self.X = np.where(X == 0, np.nan, X)

        # differentiate between all possible home slots and those feasible
        self.sets_home = input.sets["home"]
        self.sets_home_feasible = self._get_feasible_home_slots(params.r_max)

        # initialize transportation object
        self.tps = TPS(
            sets_forbidden=input.sets["forbidden"],
            sets_home=self.sets_home_feasible,
            m=params.m,
            p=params.p,
            r_max=params.r_max,
            penalties=params.penalties,
        )

        # initialize perturbation object
        self.perturbation = Perturbation(alpha=params.alpha, beta=params.beta)

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

        # NOTE: Home costs don't take into account later assigned games but the tabu
        # phase will account for it - however there is a slight risk that a new best
        # between the downward biased starting point and the actual cost is missed;
        # there is always a slight delay between the actual cost and the reported cost

        # pick best method to set schedule after construction phase
        if cost1 < cost2:
            self.logger.info("Initialization method 1 is best")
            self.X, self.list_home_costs = X1, list_home_costs1
        else:
            self.logger.info("Initialization method 2 is best")
            self.X, self.list_home_costs = X2, list_home_costs2

    def tabu_phase(
        self, progress_bar: st.delta_generator.DeltaGenerator = None
    ) -> None:
        """
        Solves transportation problem to (re)schedule all home games of
        a non-tabu team (= not recently chosen), for a certain number of
        iterations or until the full cost reaches zero. Every new optimal
        schedule is added to self.X.

        Note that this implementation nowhere enforces a minimal cost change
        before allowed to continue to the next iteration.

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
                # perturbate if no better solution found for a while, else keep going
                if (
                    self.list_full_costs[-1]
                    >= self.list_full_costs[-self.perturbation_length]
                ):
                    n_unsched_pre = np.sum(np.isnan(X), axis=1)
                    self.logger.info(f"Perturbating schedule at iteration {it}")
                    self.perturbation.perturbate(X)

                    # adjust costs based on dropped games from perturbation
                    n_unsched_pos = np.sum(np.isnan(X), axis=1)
                    list_home_costs += (n_unsched_pos - n_unsched_pre) * self.tps.p

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
                self.X = X.copy()  # update to new optimal schedule

        # set progress bar to 100% (needed in case of early termination)
        if progress_bar is not None:
            progress_bar.progress(1.0)

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

    def validate_calendar(
        self, df: pd.DataFrame, fl_net_rest_days: bool = False
    ) -> dict:
        """Gathers a dictionary with validation data on the generated schedule."""
        d_val = {}

        # grab some general statistics first
        d_val["teams"] = len(self.input.sets["teams"])
        d_val["games"] = len(df)
        d_val["unscheduled"] = sum(df[self.output_cols[0]].isna())
        d_val["cost"] = min(self.list_full_costs)

        # overview of total number of home slots less than needed (per-team basis)
        n_req_home_games = len(self.input.sets["teams"]) - 1
        n_home_slots_short = sum(
            [
                max(n_req_home_games - len(v), 0)
                for _, v in self.sets_home_feasible.items()
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

        # overview of (adjusted) rest days in matrix form
        d_val["df_rest_days"] = self.make_df_rest_days(df, net=fl_net_rest_days)

        # overview of unused home slots
        d_val["df_unused_home_slots"] = self.make_df_unused_home_slots()

        # overview of schedules by team
        d_val["df_schedules_by_team"] = self.make_df_schedules_by_team(df)

        return d_val

    def make_df_rest_days(self, df: pd.DataFrame, net: bool = False) -> pd.DataFrame:
        """
        Forms a matrix of teams vs. number of rest days for given input schedule.

        :param df: DataFrame with generated schedule.
        :param net: If True, returns the adjusted rest days by not counting team unavailabilities as a rest day.
        """
        col_date = self.output_cols[0]
        col_home = self.output_cols[3]
        col_away = self.output_cols[4]
        col_team = "Team"

        df_teams = pd.concat(
            [
                df[[col_date, col_home]].rename(columns={col_home: col_team}),
                df[[col_date, col_away]].rename(columns={col_away: col_team}),
            ]
        ).sort_values([col_team, col_date])

        df_teams[col_date] = pd.to_datetime(df_teams[col_date])
        df_teams["n_rest_days"] = (
            df_teams.groupby(col_team)[col_date].diff().dt.days - 1
        )

        if net:
            col_out = "n_rest_days_net"
            df_teams = self._compute_rest_days_net(df_teams, col_team, col_date)
        else:
            col_out = "n_rest_days"

        df_out = df_teams.groupby(col_team)[col_out].value_counts().unstack().fillna(0)
        df_out = pd.concat(
            [df_out, pd.DataFrame(df_out.sum(axis=0), columns=["TOTAL"]).transpose()],
            axis=0,
        )

        return df_out

    def make_df_schedules_by_team(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorders the schedules input by team. Assumes it is already sorted by date and hour."""
        col_date = self.output_cols[0]
        col_home = self.output_cols[-2]
        col_away = self.output_cols[-1]
        col_team = "Team"

        # clean up some irrelevant columns from validation process first
        df.drop(columns=["pairs", "days_diff"], inplace=True)

        teams = pd.unique(df[[col_home, col_away]].values.ravel("K"))

        list_sch = []
        for team in teams:
            df_team = df[(df[col_home] == team) | (df[col_away] == team)].copy()
            df_team[col_team] = team
            list_sch.append(df_team)

        df_by_team = pd.concat(list_sch, ignore_index=True)

        # include rest days columns
        df_by_team["n_rest_days"] = (
            df_by_team.groupby(col_team)[col_date].diff().dt.days - 1
        )
        df_by_team = self._compute_rest_days_net(df_by_team, col_team, col_date)

        # sort by team and date
        df_by_team = df_by_team.sort_values(by=[col_team, col_date])

        # fix output format
        df_by_team = df_by_team.set_index(col_team)
        df_by_team[col_date] = df_by_team[col_date].dt.strftime("%d/%m/%Y")

        return df_by_team

    def make_df_unused_home_slots(self) -> pd.DataFrame:
        """Forms DataFrame with teams and their unused home slots."""
        col_date = self.output_cols[0]
        col_team = "Team"

        list_unused = []
        for team_idx, team_name in self.input.sets["teams"].items():
            unused_home_slots = sorted(
                [
                    self.input.sets["slots"][s]
                    for s in set(self.sets_home[team_idx]).difference(
                        set(self.X[team_idx, :])
                    )
                ]
            )

            df_unused = pd.DataFrame({"unused": unused_home_slots})
            if len(df_unused) == 0:
                continue
            df_unused[col_team] = team_name
            list_unused.append(df_unused)

        df_unused_all = pd.concat(list_unused)[[col_team, "unused"]].set_index(col_team)
        df_unused_all = df_unused_all.rename(columns={"unused": col_date})
        df_unused_all = df_unused_all.sort_values([col_team, self.output_cols[0]])
        df_unused_all[col_date] = df_unused_all[col_date].dt.strftime("%d/%m/%Y")

        return df_unused_all

    ##################################
    ### Plotting functionality #######
    ##################################

    def plot_minimum_costs(self, title_suffix: str = "", path: str = None) -> None:
        """
        Plots evolution of running minimum cost during tabu phase.

        :param title_suffix: Suffix to add to the title of the plot.
        :param path: Path to save the plot as an image (if not None).
        """
        if not hasattr(self, "list_full_costs"):
            self.logger.warning(
                "No costs available for plotting, run self.tabu_phase() first"
            )
            return

        list_running_minimum_cost = [
            min(self.list_full_costs[: i + 1]) for i in range(len(self.list_full_costs))
        ]

        # create plot
        plt.figure(figsize=(10, 6))
        plt.plot(list_running_minimum_cost)
        plt.title(
            f"Evolution minimum cost{(' - ' + title_suffix) if title_suffix else ''}"
        )
        plt.xlabel("Iteration")
        plt.tight_layout()

        # show or save plot
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

        plt.close()

    def plot_rest_days(
        self,
        series: pd.Series,
        clips: tuple = (3, 20),
        title_suffix: str = "",
        path: str = None,
    ) -> None:
        """
        Plots distribution of rest days between games.

        :param series: Series with number of rest days as index.
        :param clips: Tuple with lower and upper bound for clipping the series.
        :param title_suffix: Suffix to add to the title of the plot.
        :param path: Path to save the plot as an image (if not None).
        """
        series_ = series.copy()

        # clip series to a lower and upper bound
        if clips:
            lower, upper = clips
            l_name, u_name = f"<={lower}", f">={upper}"

            series_.index = series_.index.astype(int)

            bot = series_[series_.index <= lower].sum()
            top = series_[series_.index >= upper].sum()

            series_ = series_[(series_.index > lower) & (series_.index < upper)]
            series_.loc[lower], series_.loc[upper] = bot, top

            series_.rename(index={lower: l_name, upper: u_name}, inplace=True)

            index_no_lb = [idx for idx in series_.index if idx not in [l_name, u_name]]
            series_ = series_.reindex([l_name] + index_no_lb + [u_name])

            colors = [
                "skyblue" if (idx != l_name and idx != u_name) else "orange"
                for idx in series_.index
            ]

        # create plot
        plt.figure(figsize=(10, 6))
        series_.plot(kind="bar", color=colors if clips else "skyblue")
        plt.title(
            f"Distribution of rest days between games{(' - ' + title_suffix) if title_suffix else ''}"
        )
        plt.xlabel("Number of rest days")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # show or save plot
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

        plt.close()

    ##################################
    ### Class utils ##################
    ##################################

    def _get_feasible_home_slots(self, r_max: int) -> dict:
        """Returns a dictionary with feasible home slots per team."""
        # NOTE: This deals with teams providing more than one home slot within
        # 'r_max' slots by simply dropping the first slot per team
        feasible_home_slots = {}
        for team_idx, set_home in self.sets_home.items():
            feasible_home_slots[team_idx] = drop_nearby_points_from_array(
                set_home, r_max
            )
        return feasible_home_slots

    def _update_dict_available_spots(
        self,
        method: int,
        X: np.ndarray = None,
        d_spots: dict = None,
        team_idx_last: int = None,
    ) -> dict:
        """Updates available home/game spots for each team during construction phase."""
        sets_home = self.sets_home

        if team_idx_last is not None:
            d_spots.pop(team_idx_last)  # drop last processed team

        if method == 1:
            if d_spots is None:
                # initialize spots from available home time slots
                d_spots = {key: len(sets_home[key]) for key in self.input.sets["teams"]}
            else:
                # subtract current scheduled away games
                d_spots = {
                    key: v - np.sum(np.isin(sets_home[key], X[:, key]))
                    for key, v in d_spots.items()
                }
        elif method == 2:
            if d_spots is None:
                d_spots = dict.fromkeys(self.input.sets["teams"])
            for team_idx in d_spots:
                set_home = sets_home[team_idx]
                opponents = [t for t in range(X.shape[0]) if t != team_idx]

                m = self.tps.create_cost_matrix(X, team_idx, set_home, opponents)

                # count number of home slots possible for each opponent
                home_option_score = np.sum(m == 0, axis=0).min()
                d_spots[team_idx] = home_option_score

        # sort by number of spots (low to high)
        d_spots = dict(sorted(d_spots.items(), key=lambda x: x[1]))

        return d_spots

    def _get_df_forbidden(self) -> pd.DataFrame:
        """Creates DataFrame with forbidden time slots for each team."""
        if not hasattr(self, "df_forbidden"):
            col_date = self.output_cols[0]
            col_team = "Team"

            data = []
            for team_idx, time_slots in self.input.sets["forbidden"].items():
                team_name = self.input.sets["teams"][team_idx]
                for slot in time_slots:
                    date = self.input.sets["slots"][slot]
                    data.append({col_team: team_name, col_date: date})

            df_forbidden = pd.DataFrame(data)
            df_forbidden[col_date] = pd.to_datetime(df_forbidden[col_date])

            self.df_forbidden = df_forbidden

        return self.df_forbidden

    def _compute_rest_days_net(
        self, df: pd.DataFrame, col_team, col_date
    ) -> pd.DataFrame:
        """Computes net rest days for each team in the DataFrame."""
        df_forbidden = self._get_df_forbidden()

        def count_unavailable_days(
            team, start_date: pd.Timestamp, end_date: pd.Timestamp
        ) -> int:
            """Counts number of unavailable days for a team within interval."""
            mask = (
                (df_forbidden[col_team] == team)
                & (df_forbidden[col_date] > start_date)
                & (df_forbidden[col_date] < end_date)
            )
            return df_forbidden[mask].shape[0]

        df["n_rest_days_net"] = df.apply(
            lambda row: (
                row["n_rest_days"]
                - count_unavailable_days(
                    row[col_team],
                    # original start date
                    row[col_date] - pd.Timedelta(days=row["n_rest_days"] + 1),
                    row[col_date],
                )
                if pd.notnull(row["n_rest_days"])
                else None
            ),
            axis=1,
        )

        return df
