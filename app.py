from datetime import datetime
from time import time

import pandas as pd
import streamlit as st

from leaguescheduler import InputParser, LeagueScheduler, SchedulerParams
from leaguescheduler.constants import MAX_ALLOWED_REST_DAYS, OUTPUT_COLS
from leaguescheduler.utils import download_output, gather_stats

DEFAULTS = SchedulerParams()
SHOW_TOP = False

st.set_page_config(page_title="League Scheduler", page_icon="⚽📅", layout="wide")

st.markdown("## 📅 League Scheduler")

row0 = st.container()
header_col_1, header_col_2 = row0.columns([0.925, 0.075])

header_col_1.markdown("#### Easily Schedule Double Round-Robin (2RR) Leagues")
go = header_col_2.button("Schedule")

row1 = st.container()
main_col1, main_col2, main_col3 = row1.columns([0.425, 0.35, 0.225])

# variables that require to be set upfront
sheet_names = []
d_penalties = {}

with main_col1:
    st.markdown("**Input file**")

    st.info(
        "One league per sheet, use **NIET** for unavailability ([example](https://github.com/sborms/leaguescheduler/blob/main/example_input.xlsx))",
        icon="ℹ️",
    )

    file = st.file_uploader(
        "Provide input Excel sheet",
        type=["xlsx"],
    )

    if file is not None:
        input = InputParser(file)
        sheet_names = input.sheet_names
        d_penalties = input.penalties

    # sheet names as checkboxes
    n_sheet_cols, selected_sheets = 4, []
    columns = st.columns(n_sheet_cols)
    for sheet_idx, sheet_name in enumerate(sheet_names):
        if columns[sheet_idx % n_sheet_cols].checkbox(sheet_name, value=True):
            selected_sheets.append(sheet_name)

with main_col2:
    st.markdown("**Parameters**")

    st.info(
        "See `help(SchedulerParams)` for full parametrization",
        icon="ℹ️",
    )

    main_col2_sub_col1, main_col2_sub_col2, main_col2_sub_col3 = main_col2.columns(
        [1, 1, 1]
    )
    m = main_col2_sub_col1.number_input(
        "**Rest days pairs** (`m - 1`)",
        min_value=0,  # m = 1
        max_value=100,
        value=DEFAULTS.m - 1,
    )
    r_max = main_col2_sub_col2.number_input(
        "**Rest days** (`r_max - 2`)",
        min_value=0,  # r_max = 2
        max_value=20,
        value=DEFAULTS.r_max - 2,
    )
    n_iterations = main_col2_sub_col3.number_input(
        "**Maximum iterations**",
        min_value=10,
        max_value=50000,
        value=DEFAULTS.n_iterations,
    )

    main_col2_sub_col3, main_col2_sub_col4 = main_col2.columns([1, 1])
    unscheduled_date = main_col2_sub_col3.text_input(
        "**Unscheduled date**",
        value="31/07/2025",
    )

    unscheduled_hour = main_col2_sub_col4.text_input(
        "**Unscheduled hour**",
        value="00u",
    )

main_col3.markdown("**Penalties**", unsafe_allow_html=True)
main_col3.info(
    "Provide as **penalties** sheet in input",
    icon="📌",
)

if not d_penalties:
    main_col3.markdown("No penalties input")
else:
    df_penalties = pd.DataFrame.from_dict(
        d_penalties, orient="index", columns=["Penalty"]
    ).rename_axis("Rest day")
    df_penalties.index = df_penalties.index - 1  # rest days = n_days - 1 for display
    main_col3.dataframe(df_penalties, use_container_width=True, height=5 * 35 + 3)

st.markdown("---")

output_col1, output_col2 = st.columns([1, 3], gap="medium")

with output_col1:
    # perform scheduling
    output_sch, output_res, output_unu, output_tea = {}, {}, {}, {}
    if go:
        if file is None:
            st.markdown("Upload a file first!")
        else:
            start_time = time()

            d_stats = None
            index_table = []

            for sheet_name in selected_sheets:
                st.markdown(f"Scheduling league **{sheet_name}**")

                input.from_excel(sheet_name=sheet_name)
                input.parse()

                params = SchedulerParams(
                    n_iterations=n_iterations,
                    m=m + 1,  # from rest days to time slots
                    r_max=r_max + 2,  # from rest days to time slots
                    penalties=d_penalties,
                )

                scheduler = LeagueScheduler(
                    input,
                    params=params,
                )

                scheduler.construction_phase()

                progress_bar = st.progress(0.0)
                scheduler.tabu_phase(progress_bar)

                # create calendar
                df = scheduler.create_calendar()

                # compute validation statistics
                d_val = scheduler.validate_calendar(df, fl_net_rest_days=True)
                d_stats = gather_stats(d_val, d_stats)
                index_table += [sheet_name]

                # gather info for top calendars
                if SHOW_TOP:
                    top_X, list_all_calendars = scheduler.top_X, []
                    # NOTE: Drop first one because this is the optimal solution
                    for _, d_X in enumerate(top_X[1:], start=1):
                        cost, X = d_X["cost"], d_X["X"]

                        _df = scheduler.create_calendar(X)
                        list_all_calendars.append(_df)

                        d_val = scheduler.validate_calendar(
                            _df, fl_net_rest_days=True, cost=cost
                        )
                        d_stats = gather_stats(d_val, d_stats)
                    index_table += [sheet_name] * (len(top_X) - 1)

                # store calendar output
                df_out = df[OUTPUT_COLS].copy()
                df_out[OUTPUT_COLS[0]] = df_out[OUTPUT_COLS[0]].dt.strftime("%d/%m/%Y")
                mask_unscheduled = df_out[OUTPUT_COLS[0]].isna()
                df_out.loc[mask_unscheduled, OUTPUT_COLS[0]] = unscheduled_date
                df_out.loc[mask_unscheduled, OUTPUT_COLS[1]] = unscheduled_hour
                df_out.reset_index(drop=True, inplace=True)
                output_sch[sheet_name] = df_out

                # store rest days output
                output_res[sheet_name] = d_val["df_rest_days"]

                # store unused home slots output
                output_unu[sheet_name] = d_val["df_unused_home_slots"]

                # store schedules by team
                output_tea[sheet_name] = d_val["df_schedules_by_team"]

            elapsed_time = time() - start_time
            st.markdown(f"**Done!** Took {int(elapsed_time)} seconds.")

with output_col2:
    if len(output_sch) > 0:
        df_stats = pd.DataFrame(d_stats, index=index_table)
        df_stats = df_stats.astype(int)

        # remove cost of unfeasible schedules
        df_stats["cost"] = df_stats["cost"] - (
            df_stats["missing_home_slots"] * DEFAULTS.p
        )

        # show summary table
        st.markdown("**Summary**")
        st.markdown(
            f"The last two columns show how often the **actual number of rest days** exceeds {MAX_ALLOWED_REST_DAYS}, including (`_all`) or excluding (`_rel`) teams without available home slots. "
            "The algorithm itself **does not penalize** excessive rest days for those teams, hence the latter column is most _relevant_. The fact that the table displays the actual rest days and not the adjusted "
            "ones accounting for unavailability, makes it 'worst-case' indicators."
        )
        st.table(df_stats)

        # download output file with schedules and additional info
        st.download_button(
            label="Download",
            data=download_output(
                output_sch, output_tea, output_unu, output_res, df_stats
            ),
            file_name=f"{'_'.join(selected_sheets)}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown(
            "_Once you download the schedules, the shown output will disappear._"
        )
