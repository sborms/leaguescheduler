import io
import logging

import numpy as np
import pandas as pd
import streamlit as st


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
            "max_rest_days",
            "n_high_rest_days_all",
            "n_high_rest_days_rel",
        ]
        d_stats = {key: [] for key in keys}

    for key in d_stats.keys():
        d_stats[key].append(d_val[key])

    return d_stats


def download_output(
    output_sch: dict,
    output_tea: dict,
    output_unu: dict,
    output_res: dict,
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
            output_res[sheet].to_excel(writer, sheet_name=f"{sheet}_rest", index=True)
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


def drop_nearby_points_from_array(arr: np.array, r_max: int) -> np.array:
    """Sequentially drops second point if pairwise difference is less than r_max."""
    if len(arr) <= 1:
        return arr.copy()

    arr_out = [arr[0]]
    for i in range(1, len(arr)):
        val = arr[i]
        if (val - arr_out[-1]) + 1 < r_max:
            continue
        else:
            arr_out.append(val)

    return np.array(arr_out)
