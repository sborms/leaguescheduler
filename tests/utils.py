import numpy as np

from leaguescheduler import InputParser, LeagueScheduler, SchedulerParams


def optimize(
    n_teams: int,
    n_iterations: int,
    input_file: str = "example/input.xlsx",
    sheet_name: str = "LEAGUE A",
) -> np.ndarray:
    input = InputParser(input_file)

    input.from_excel(sheet_name=sheet_name)
    input.parse()

    input.data = input.data.iloc[:, : n_teams + 1]

    scheduler = LeagueScheduler(
        input=input,
        params=SchedulerParams(n_iterations=n_iterations, penalties=input.penalties),
    )
    scheduler.construction_phase()
    scheduler.tabu_phase()

    return scheduler.X
