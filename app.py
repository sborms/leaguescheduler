from time import time

import pandas as pd
import streamlit as st

from leaguescheduler import InputParser, LeagueScheduler
from leaguescheduler.constants import OUTPUT_COLS
from leaguescheduler.utils import download_output, gather_stats, penalty_input

P = 5000

st.set_page_config(page_title="League Scheduler", page_icon="⚽", layout="wide")

st.title("League Scheduler")
st.markdown("#### Schedule your double round-robin leagues with ease")

row1 = st.container()
main_col1, main_col2, main_col3, main_col4, main_col5 = row1.columns([3, 2, 1, 1, 1])

with main_col1:
    st.markdown("**Input file**")

    file = st.file_uploader(
        "Upload one league input per sheet (use **NIET** for unavailability)",
        type=["xlsx"],
    )

    sheet_names = []
    if file is not None:
        input = InputParser(file)
        sheet_names = input.sheet_names

    # sheet names as checkboxes
    n_sheet_cols, selected_sheets = 3, []
    columns = st.columns(n_sheet_cols)
    for sheet_idx, sheet_name in enumerate(sheet_names):
        if columns[sheet_idx % n_sheet_cols].checkbox(sheet_name, value=True):
            selected_sheets.append(sheet_name)

with main_col2:
    st.markdown("**Parameters**")

    m = st.number_input(
        "**Min. rest days between pairs of games**",
        min_value=0,
        max_value=100,
        value=6,
    )
    R_max = st.number_input(
        "**Required days for 2 games of same team**",
        min_value=1,
        max_value=20,
        value=5,
    )
    n_iterations = st.number_input(
        "**Max. number of iterations**",
        min_value=10,
        max_value=50000,
        value=10000,
    )

    main_col2_sub_col1, main_col2_sub_col2 = main_col2.columns([1, 1])
    unscheduled_date = main_col2_sub_col1.text_input(
        "**Unscheduled date**",
        value="31/07/2024",
    )

    unscheduled_hour = main_col2_sub_col2.text_input(
        "**Unscheduled hour**",
        value="00u",
    )

main_col3.markdown("**Penalties**", unsafe_allow_html=True)
p1 = penalty_input(main_col3, "0", 1000)
p2 = penalty_input(main_col3, "1 & >=29 ", 400)
p3 = penalty_input(main_col3, "2 & 23-28", 160)

main_col4.markdown("&nbsp;", unsafe_allow_html=True)
p4 = penalty_input(main_col4, "3 & 19-22", 64)
p5 = penalty_input(main_col4, "4 & 18", 26)
p6 = penalty_input(main_col4, "5 & 17", 10)

main_col5.markdown("&nbsp;", unsafe_allow_html=True)
p7 = penalty_input(main_col5, "6 & 16", 4)
p8 = penalty_input(main_col5, "7 & 15", 2)
p9 = penalty_input(main_col5, "8 & 14", 1)

st.markdown("---")

output_col1, output_col2 = st.columns([2, 3])

with output_col1:
    go = st.button("Schedule")

    # perform scheduling
    output_sch, output_val, output_unu = {}, {}, {}
    if go:
        if file is None:
            st.markdown("Upload a file first!")
        else:
            start_time = time()

            d_stats = None
            input = InputParser(file)

            # gather penalties incl. those for >=29 rest days (~ up to 60)
            d_penalties = {
                1: p1,
                2: p2,
                3: p3,
                24: p3,
                25: p3,
                26: p3,
                27: p3,
                28: p3,
                29: p3,
                4: p4,
                20: p4,
                21: p4,
                22: p4,
                23: p4,
                5: p5,
                19: p5,
                6: p6,
                18: p6,
                7: p7,
                17: p7,
                8: p8,
                16: p8,
                9: p9,
                15: p9,
            }

            d_penalties_above_29 = {k: p2 for k in range(30, 61)}
            d_penalties.update(d_penalties_above_29)

            for sheet_name in selected_sheets:
                st.markdown(f"Scheduling league **{sheet_name}**")

                input.read(sheet_name=sheet_name)
                input.parse()

                scheduler = LeagueScheduler(
                    input,
                    P=P,
                    n_iterations=n_iterations,
                    m=m + 1,  # from rest days to time slots
                    R_max=R_max,
                    penalties=d_penalties,
                )

                scheduler.construction_phase()

                progress_bar = st.progress(0.0)
                scheduler.tabu_phase(progress_bar)

                # create calendar
                df = scheduler.create_calendar()

                # compute validation statistics
                d_val = scheduler.validate_calendar(df)
                d_stats = gather_stats(d_val, d_stats)

                # store calendar output
                df_out = df[OUTPUT_COLS].copy()
                df_out[OUTPUT_COLS[0]] = df_out[OUTPUT_COLS[0]].dt.strftime("%d/%m/%Y")
                mask_unscheduled = df_out[OUTPUT_COLS[0]].isna()
                df_out.loc[mask_unscheduled, OUTPUT_COLS[0]] = unscheduled_date
                df_out.loc[mask_unscheduled, OUTPUT_COLS[1]] = unscheduled_hour
                df_out.reset_index(drop=True, inplace=True)
                output_sch[sheet_name] = df_out

                # store rest days output
                output_val[sheet_name] = d_val["df_rest_days"]

                # store unused home slots output
                output_unu[sheet_name] = d_val["df_unused_home_slots"]

        elapsed_time = time() - start_time
        st.markdown(f"**Done!** Took {int(elapsed_time)} seconds.")

with output_col2:
    if len(output_sch) > 0:
        df_stats = pd.DataFrame(d_stats, index=selected_sheets)
        df_stats = df_stats.astype(int)

        # remove cost of unfeasible schedules
        df_stats["cost"] = df_stats["cost"] - (df_stats["missing_home_slots"] * P)

        # download output file with schedules and additional info
        st.download_button(
            label="Download",
            data=download_output(output_sch, output_val, output_unu, df_stats),
            file_name="schedules.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("**Summary**")
        st.table(df_stats)

        st.markdown(
            "_Once you download the schedules, the output here will disappear._"
        )
