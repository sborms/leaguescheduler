from time import time

import pandas as pd
import streamlit as st

from leaguescheduler import InputParser, LeagueScheduler
from leaguescheduler.constants import OUTPUT_COLS
from leaguescheduler.utils import download_output, gather_stats, penalty_input

st.set_page_config(page_title="League Scheduler", page_icon="⚽", layout="wide")

st.title("League Scheduler")
st.markdown("#### Schedule your double round-robin leagues with ease")

row1 = st.container()
main_col1, main_col2, main_col3, main_col4, main_col5 = row1.columns([3, 2, 1, 1, 1])

with main_col1:
    st.markdown("Input file")

    file = st.file_uploader(
        "Upload one league input per sheet (use **NIET** for unavailability).",
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
    st.markdown("Parameters")

    m = st.number_input(
        "**Min. days between pairs of games**",
        min_value=0,
        max_value=100,
        value=30,
    )
    R_max = st.number_input(
        "**Min. slots for >1 game of same team**",
        min_value=1,
        max_value=10,
        value=4,
    )
    n_iterations = st.number_input(
        "**Max. number of iterations**",
        min_value=10,
        max_value=2000,
        value=1000,
    )

main_col3.markdown("_Penalties_", unsafe_allow_html=True)
p1 = penalty_input(main_col3, "0", 750)
p2 = penalty_input(main_col3, "1", 500)
p3 = penalty_input(main_col3, "2", 250)

main_col4.markdown("&nbsp;", unsafe_allow_html=True)
p4 = penalty_input(main_col4, "3", 100)
p5 = penalty_input(main_col4, "4 & 18", 50)
p6 = penalty_input(main_col4, "5 & 17", 30)

main_col5.markdown("&nbsp;", unsafe_allow_html=True)
p7 = penalty_input(main_col5, "6 & 16", 20)
p8 = penalty_input(main_col5, "7 & 15", 15)
p9 = penalty_input(main_col5, "8 & 14", 10)

st.markdown("---")

output_col1, output_col2 = st.columns(2)

with output_col1:
    go = st.button("Schedule")

    # perform scheduling
    output_sch, output_val = {}, {}
    if go:
        if file is None:
            st.markdown("Upload a file first!")
        else:
            start_time = time()

            d_stats = None
            input = InputParser(file)

            for sheet_name in selected_sheets:
                st.markdown(f"Scheduling league **{sheet_name}**")

                input.read(sheet_name=sheet_name)
                input.parse()

                scheduler = LeagueScheduler(
                    input,
                    P=5000,
                    n_iterations=n_iterations,
                    m=m,
                    R_max=R_max,
                    # penalties={k + 1: v for k, v in DICT_REST_DAYS.items()},
                    penalties={
                        1: p1,
                        2: p2,
                        3: p3,
                        4: p4,
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
                    },
                )

                scheduler.construction_phase()

                progress_bar = st.progress(0.0)
                scheduler.tabu_phase(progress_bar)

                # create calendar
                df = scheduler.create_calendar()

                # compute validation statistics
                d_val = scheduler.validate(df)
                d_stats = gather_stats(d_val, d_stats)

                # store output
                output_sch[sheet_name] = df[OUTPUT_COLS].reset_index(drop=True)
                output_val[sheet_name] = d_val["df_rest_days"]

        elapsed_time = time() - start_time
        st.markdown(f"**Done!** Took {int(elapsed_time)} seconds.")

with output_col2:
    if len(output_sch) > 0:
        df_stats = pd.DataFrame(d_stats, index=selected_sheets)
        df_stats = df_stats.astype(int)

        # download output file with schedules
        st.download_button(
            label="Download",
            data=download_output(output_sch, output_val, df_stats),
            file_name="schedules.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("**Summary**")
        st.dataframe(df_stats, use_container_width=True)

        st.markdown(
            "_Once you download the schedules, the output here will disappear._"
        )
