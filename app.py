from time import time

import pandas as pd
import streamlit as st

from leaguescheduler import InputParser, LeagueScheduler
from leaguescheduler.constants import OUTPUT_COLS
from leaguescheduler.utils import download_output, gather_stats, penalty_input

UNSCHEDULED_DATE = "31/07/{year}"
UNSCHEDULED_HOUR = "00u"

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
        value=7,
    )
    R_max = st.number_input(
        "**Min. slots for 2 games of same team**",
        min_value=1,
        max_value=20,
        value=10,
    )
    n_iterations = st.number_input(
        "**Max. number of iterations**",
        min_value=10,
        max_value=50000,
        value=5000,
    )

main_col3.markdown("_Penalties_", unsafe_allow_html=True)
p1 = penalty_input(main_col3, "0", 1000)
p2 = penalty_input(main_col3, "1", 400)
p3 = penalty_input(main_col3, "2", 160)

main_col4.markdown("&nbsp;", unsafe_allow_html=True)
p4 = penalty_input(main_col4, "3 & >=19", 64)
p5 = penalty_input(main_col4, "4 & 18", 26)
p6 = penalty_input(main_col4, "5 & 17", 10)

main_col5.markdown("&nbsp;", unsafe_allow_html=True)
p7 = penalty_input(main_col5, "6 & 16", 4)
p8 = penalty_input(main_col5, "7 & 15", 2)
p9 = penalty_input(main_col5, "8 & 14", 1)

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

            # gather penalties incl. those for >=19 rest days (~ up to 50)
            d_penalties = {
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
            }

            d_penalties_above_19 = {k: p4 for k in range(20, 51)}
            d_penalties.update(d_penalties_above_19)

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
                max_year = int(
                    df_out[OUTPUT_COLS[0]].dt.year.max()
                )  # TODO: Make this more general?
                df_out[OUTPUT_COLS[0]] = df_out[OUTPUT_COLS[0]].dt.strftime("%d/%m/%Y")
                mask_unscheduled = df_out[OUTPUT_COLS[0]].isna()
                df_out.loc[mask_unscheduled, OUTPUT_COLS[0]] = UNSCHEDULED_DATE.format(
                    year=max_year
                )
                df_out.loc[mask_unscheduled, OUTPUT_COLS[1]] = UNSCHEDULED_HOUR
                df_out.reset_index(drop=True, inplace=True)
                output_sch[sheet_name] = df_out

                # store rest days output
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
            f"Unscheduled games are marked with **{UNSCHEDULED_DATE.format(year=max_year)} {UNSCHEDULED_HOUR}**."
        )
        st.markdown(
            "_Once you download the schedules, the output here will disappear._"
        )
