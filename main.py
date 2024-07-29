import json
import os

import click
import numpy as np
import pandas as pd

from leaguescheduler import InputParser, LeagueScheduler
from leaguescheduler.utils import gather_stats, setup_logger


# fmt: off
@click.command()
@click.option("--config_file", default=None, help="Path to a configuration JSON file with (part of) the arguments.")
@click.option("--input_file", help="Input Excel file with for every team their (in)availability data.")
@click.option("--output_folder", help="Folder where the outputs (logs, overview, schedules) will be stored.")
@click.option("--seed", default=None, type=int, help="Optional seed for np.random.seed().")
@click.option("--tabu_length", default=4, type=int, help="Number of iterations during which a team cannot be selected.")
@click.option("--perturbation_length", default=50, type=int, help="Check perturbation need every this many iterations.")
@click.option("--n_iterations", default=1000, type=int, help="Number of tabu phase iterations.")
@click.option("--m", default=14, type=int, help="Minimum number of time slots between 2 games with same pair of teams.")
@click.option("--p", default=5000, type=int, help="Cost from dummy supply node q to non-dummy demand node.")  # P
@click.option("--r_max", default=4, type=int, help="Minimum required time slots for 2 games of same team.")  # R_max
@click.option("--penalties", default={1: 10, 2: 3, 3: 1}, type=dict, help="Dictionary as {n_days: penalty} where n_days ~ rest days + 1.")
@click.option("--alpha", default=0.50, type=float, help="Picks perturbation operator 1 with probability alpha.")
@click.option("--beta", default=0.01, type=float, help="Probability of removing a game in operator 1.")
@click.option("--unavailable", default="NIET", type=str, help="Cell value to indicate that a team is unavailable.")
@click.option("--clip_bot", default=2, type=int, help="Value for clipping rest days plot on low end.")  # clips[0]
@click.option("--clip_top", default=20, type=int, help="Value for clipping rest days plot on high end.")  # clips[1]
@click.pass_context
# fmt: on
def main(ctx, config_file, **kwargs):
    ############ start: argument parsing ############
    if config_file is not None:
        with open(config_file, "r") as f:
            config = json.load(f)

    # use command-line argument if provided, else use config value or default
    def get_value(key, default=None):
        if (
            config_file is None
            or ctx.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE
        ):
            return kwargs[key]  # command-line argument or click default
        else:
            return config.get(key, default)  # config value or default

    input_file = get_value("input_file")
    output_folder = get_value("output_folder")
    seed = get_value("seed")
    tabu_length = get_value("tabu_length", 4)
    perturbation_length = get_value("perturbation_length", 50)
    n_iterations = get_value("n_iterations", 1000)
    m = get_value("m", 14)
    p = get_value("p", 5000)
    r_max = get_value("r_max", 4)
    penalties = get_value("penalties", {1: 10, 2: 3, 3: 1})
    alpha = get_value("alpha", 0.50)
    beta = get_value("beta", 0.01)
    unavailable = get_value("unavailable", "NIET")
    clip_bot = get_value("clip_bot", 2)
    clip_top = get_value("clip_top", 20)

    # ensure keys of penalties are integers
    if penalties:
        penalties = {int(k): v for k, v in penalties.items()}
    ############ end: argument parsing ############

    if seed is not None:
        np.random.seed(seed)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        click.secho(f"Created output folder {output_folder}/", fg="green")

    logger = setup_logger(logfile=f"{output_folder}/logs.log")

    logger.info("Overview of input arguments:")
    print(locals().keys())
    for key, value in locals().items():
        if key not in [
            "ctx",
            "config_file",
            "kwargs",
            "f",
            "config",
            "get_value",
            "logger",
        ]:
            logger.info(f" --> {key} = {value}")

    d_stats = None
    input = InputParser(input_file, unavailable=unavailable)

    for sheet_name in input.sheet_names:
        logger.info(f"PROCESSING LEAGUE > {sheet_name}")

        input.read(sheet_name=sheet_name)
        input.parse()
        logger.info("Read and parsed input data")

        scheduler = LeagueScheduler(
            input=input,
            tabu_length=tabu_length,
            perturbation_length=perturbation_length,
            n_iterations=n_iterations,
            m=m,
            P=p,
            R_max=r_max,
            penalties=penalties,
            alpha=alpha,
            beta=beta,
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
            clips=(clip_bot, clip_top),
            title_suffix=sheet_name,
            path=f"{output_folder}/{sheet_name}_rest_days.png",
        )
        logger.info("Stored rest days plot")

    df_stats = pd.DataFrame(d_stats, index=input.sheet_names)
    df_stats.to_excel(f"{output_folder}/stats.xlsx")

    click.secho(f"Done!", fg="green")


if __name__ == "__main__":
    main()
