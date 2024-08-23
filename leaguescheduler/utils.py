import io
import logging

import pandas as pd
import streamlit as st

from .constants import OUTPUT_COLS


def ndigits(number: float) -> int:
    """Returns number of digits in a number."""
    return len(str(number))


def fill_value(x, unavailable: str) -> int:
    """Returns -1 if 'unavailable', 0 if NaN, and 1 if any other string."""
    if x == unavailable:
        return -1
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        return 1


def setup_logger(logfile: str) -> logging.Logger:
    """Sets up logger with file and console/terminal handlers."""
    logger = logging.getLogger(__name__)

    # remove existing handlers first
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def gather_stats(d_val: dict, d_stats: dict = None) -> dict:
    """Updates statistics dictionary from validation dictionary."""
    if d_stats is None:
        keys = [
            "cost",
            "teams",
            "games",
            "unscheduled",
            "missing_home_slots",
            "min_gap_pairs",
            # "max_gap_pairs",
        ]
        d_stats = {key: [] for key in keys}

    for key in d_stats.keys():
        d_stats[key].append(d_val[key])

    return d_stats


def get_schedules_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """Reorders the schedules input by team. Assumes it is already sorted by date and hour."""
    col_date = OUTPUT_COLS[0]
    col_home = OUTPUT_COLS[-2]
    col_away = OUTPUT_COLS[-1]
    col_team = "Team"

    teams = pd.unique(df[[col_home, col_away]].values.ravel("K"))

    list_sch = []
    for team in teams:
        df_team = df[(df[col_home] == team) | (df[col_away] == team)].copy()
        df_team[col_team] = team
        list_sch.append(df_team)

    df_by_team = pd.concat(list_sch, ignore_index=True)
    df_by_team.set_index(col_team, inplace=True)

    # sort by team and date
    df_by_team["date_to_sort"] = pd.to_datetime(df_by_team[col_date], format="%d/%m/%Y")
    df_by_team_sorted = df_by_team.sort_values(by=[col_team, "date_to_sort"]).drop(
        columns="date_to_sort"
    )

    return df_by_team_sorted


def download_output(
    output_sch: dict,
    output_tea: dict,
    output_unu: dict,
    output_val: dict,
    df_stats: pd.DataFrame,
) -> io.BytesIO:
    """Writes outputs to Excel file without storing it locally."""
    file = io.BytesIO()
    with pd.ExcelWriter(file, engine="xlsxwriter") as writer:
        df_stats.to_excel(writer, sheet_name="overview", index=True)
        for sheet in output_sch.keys():  # assumes keys of inputs match
            # add schedule sheet and fix date format in Excel
            output_sch[sheet].to_excel(writer, sheet_name=sheet, index=False)

            workbook = writer.book
            date_format = workbook.add_format({"num_format": "dd/mm/yyyy"})
            writer.sheets[sheet].set_column(0, 0, None, date_format)

            # add other sheets
            output_tea[sheet].to_excel(writer, sheet_name=f"{sheet}_teams", index=True)
            output_unu[sheet].to_excel(writer, sheet_name=f"{sheet}_unused", index=True)
            output_val[sheet].to_excel(writer, sheet_name=f"{sheet}_rest", index=True)
    file.seek(0)
    return file


def penalty_input(col: st.columns, n: str, value: int) -> st.number_input:
    """Returns a penalty Streamlit input field for given column."""
    return col.number_input(
        f"**{n} rest {'day' if n == '1' else 'days'}**",
        min_value=0,
        max_value=1000,
        value=value,
    )
