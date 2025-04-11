import os
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich import print

from leaguescheduler import InputParser, LeagueScheduler, SchedulerParams
from leaguescheduler.utils import gather_stats, setup_logger

# NOTE: This approach https://gist.github.com/tbenthompson/9db0452445451767b59f5cb0611ab483 allows to use an overridable config file
# NOTE: Underscores are replaced with dashes in the command line arguments
# NOTE: Documentation of the SchedulerParams is copy-pasted from the dataclass

app = typer.Typer()


# fmt: off
@app.command()
def main(
    input_file: Annotated[str, typer.Option(help="Input Excel file with for every team their (un)availability data.")],
    output_folder: Annotated[str, typer.Option(help="Folder where the outputs (logs, overview, schedules) will be stored.")],
    seed: Annotated[int, typer.Option(help="Optional seed for np.random.seed().")] = None,
    unavailable:  Annotated[str, typer.Option(help="Cell value to indicate that a team is unavailable.")] = "NIET",
    clip_bot: Annotated[int, typer.Option(help="Value for clipping rest days plot on low end.")] = 2,  # clips[0]
    clip_upp: Annotated[int, typer.Option(help="Value for clipping rest days plot on high end.")] = 20,  # clips[1]
    tabu_length: Annotated[int, typer.Option(help="Number of iterations during which a team cannot be selected.")] = SchedulerParams.tabu_length,
    perturbation_length: Annotated[int, typer.Option(help="Check perturbation need every this many iterations.")] = SchedulerParams.perturbation_length,
    n_iterations: Annotated[int, typer.Option(help="Number of tabu phase iterations.")] = SchedulerParams.n_iterations,
    m: Annotated[int, typer.Option(help="Minimum number of time slots between 2 games with same pair of teams.")] = SchedulerParams.m,
    p: Annotated[int, typer.Option(help="Cost from dummy supply node q to non-dummy demand node.")] = SchedulerParams.P,
    r_max: Annotated[int, typer.Option(help="Minimum required time slots for 2 games of same team.")] = SchedulerParams.R_max,
    alpha: Annotated[float, typer.Option(help="Probability of picking perturbation operator 1.")] = SchedulerParams.alpha,
    beta: Annotated[float, typer.Option(help="Probability of removing a game in operator 1.")] = SchedulerParams.beta,
):
    # fmt: on
    if seed is not None:
        np.random.seed(seed)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[green]Created output folder {output_folder}/[/green]")

    logger = setup_logger(logfile=f"{output_folder}/logs.log")

    logger.info("Inputs:")
    for key, value in locals().items():
        if key != "logger":
            logger.info(f" --> {key} = {value}")

    d_stats = None
    input = InputParser(input_file, unavailable=unavailable)
    penalties = input.penalties

    for sheet_name in input.sheet_names:
        logger.info(f"PROCESSING LEAGUE > {sheet_name}")

        input.from_excel(sheet_name=sheet_name)
        input.parse()
        logger.info("Read and parsed input data")

        params = SchedulerParams(
            tabu_length=tabu_length,
            perturbation_length=perturbation_length,
            n_iterations=n_iterations,
            m=m,
            P=p,
            R_max=r_max,
            penalties=penalties,
            alpha=alpha,
            beta=beta,
        )

        scheduler = LeagueScheduler(
            input=input,
            params=params,
            logger=logger,
        )

        logger.info("Phase 1: Construction")
        scheduler.construction_phase()

        logger.info("Phase 2: Tabu & perturbation")
        scheduler.tabu_phase()
        logger.info("Completed scheduling")

        scheduler.plot_minimum_costs(
            title_suffix=sheet_name, path=f"{output_folder}/{sheet_name}_costs.png"
        )
        logger.info("Stored running minimum cost plot")

        df = scheduler.create_calendar()
        scheduler.store_calendar(df, file=f"{output_folder}/{sheet_name}.xlsx")
        logger.info("Stored calendar")

        d_val = scheduler.validate_calendar(df)
        d_stats = gather_stats(d_val, d_stats)
        logger.info("Gathered validation info")

        scheduler.plot_rest_days(
            series=d_val["df_rest_days"].loc["TOTAL"],
            clips=(clip_bot, clip_upp),
            title_suffix=sheet_name,
            path=f"{output_folder}/{sheet_name}_rest_days.png",
        )
        logger.info("Stored rest days plot")

    df_stats = pd.DataFrame(d_stats, index=input.sheet_names)
    df_stats.to_excel(f"{output_folder}/stats.xlsx")

    print("[green]Done![/green] :smile:")
