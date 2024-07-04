import os

import click
import numpy as np
import pandas as pd

from leaguescheduler import InputParser, LeagueScheduler
from leaguescheduler.utils import gather_stats, setup_logger


# fmt: off
@click.command()
@click.option("--file", help="Input Excel file with for every team their (in)availability data.")
@click.option("--output_folder", help="Folder where the outputs (logs, overview, schedules) will be stored.")
@click.option("--seed", default=None, type=int, help="Optional seed for np.random.seed().")
@click.option("--tabu_length", default=4, type=int, help="Number of iterations during which a team cannot be selected.")
@click.option("--perturbation_length", default=50, type=int, help="Check perturbation need every this many iterations.")
@click.option("--n_iterations", default=1500, type=int, help="Number of tabu phase iterations.")
@click.option("--m", default=14, type=int, help="Minimum number of time slots between 2 games with same pair of teams.")
@click.option("--p", default=5000, type=int, help="Cost from dummy supply node q to non-dummy demand node.")  # P
@click.option("--r_max", default=4, type=int, help="Ideal minimum time slots for 2 games of same team.")  # R_max
@click.option("--penalties", default={1: 10, 2: 3, 3: 1}, type=dict, help="Dictionary as {n_days: penalty} where n_days ~ rest days + 1.")
@click.option("--alpha", default=0.50, type=float, help="Picks perturbation operator 1 with probability alpha.")
@click.option("--beta", default=0.01, type=float, help="Probability of removing a game in operator 1.")
@click.option("--unavailable", default="NIET", type=str, help="Cell value to indicate that a team is unavailable.")
# fmt: on
def main(
    seed,
    file,
    output_folder,
    tabu_length,
    perturbation_length,
    n_iterations,
    m,
    p,
    r_max,
    penalties,
    alpha,
    beta,
    unavailable,
):
    if seed is not None:
        np.random.seed(seed)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        click.secho(f"Created output folder {output_folder}/", fg="green")

    logger = setup_logger(logfile=f"{output_folder}/logs.log")

    d_stats = None
    input = InputParser(file, unavailable=unavailable)

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

        d_val = scheduler.validate(df)
        d_stats = gather_stats(d_val, d_stats)
        logger.info("Gathered validation info")

        scheduler.plot_rest_days(
            series=d_val["df_rest_days"].loc["TOTAL"],
            clips=(2, 20),
            title_suffix=sheet_name,
            path=f"{output_folder}/{sheet_name}_rest_days.png",
        )
        logger.info("Stored rest days plot")

    df_stats = pd.DataFrame(d_stats, index=input.sheet_names)
    df_stats.to_excel(f"{output_folder}/stats.xlsx")

    click.secho(f"Done!", fg="green")


if __name__ == "__main__":
    main()
