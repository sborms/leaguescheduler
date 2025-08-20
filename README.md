# ⚽📅 2RR League Scheduler

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leaguescheduler.streamlit.app)

If you are looking to schedule a sports league with at least the following constraints...
- Everyone plays 1 home game and 1 away game against each other
- Home games are played on reserved dates
- Away games are not played on unavailable dates
- No team plays 2 games on the same day
- The calendar is spread out such that teams have enough rest days between games

... then this will help you!

This software implements **constrained time-relaxed double round-robin (2RR) sports league scheduling** using the tabu search based heuristic algorithm described in the paper [**Scheduling a non-professional indoor football league**](https://pure.tue.nl/ws/portalfiles/portal/121797609/Bulck2019_Article_SchedulingANon_professionalInd.pdf) by Van Bulck, Goosens and Spieksma (2019). The meta-algorithm heavily relies on the Hungarian algorithm as implemented in the [`munkres` package](https://software.clapper.org/munkres), to solve the transportation problem recurrently. Some tricks and specifics were added left and right, especially to minimize the occurrence of teams with an excessive number rest days (internally fixed at 28) between their games.

# Installation

Clone the repository, then install the package in **editable mode** by running the following commands in the root directory ([install `uv`](https://docs.astral.sh/uv/getting-started/installation) if necessary):

```bash
uv venv
uv pip install -e .
```

Make sure to stay in the correct environment once installed, or else use `uv run ...` for commands.

# Usage

## Input

The Excel file `example_input.xlsx` contains an example of the input data. The input should always, in the exact format as in the example, include the reserved dates (together with location and time) and unavailable dates of all teams from a single league.

One sheet corresponds to one league. 

For instance, a league could consist of 10 teams, each with about 12 to 20 reserved dates and a number of unavailable dates.

## Output

The generated output for a single league is a solutions matrix `X` that can easily be converted into a clear `DataFrame` calendar, and stored as an Excel file.

The calendar includes the date, time, location, home team and away team for each game. The unplanned games are put at the bottom.

## Scheduling

### CLI

You can use the scheduler from the command line as follows:

```bash
2rr \
--input_file "example_input.xlsx" \
--output_folder "example_output" \
--seed 321 \
--n_iterations 500
```

Alternatively, you can execute `make example` which runs the above example.

See `2rr --help` (and the research paper mentioned at the top) for more information about all the available arguments.

### Classes

To more freely play around, you can import the core classes in your own Python script or notebook.

Here's a minimal example with default parameters:

```python
from leaguescheduler import InputParser, LeagueScheduler

input = InputParser(input_file)
input.from_excel(sheet_name=input.sheet_names[0])
input.parse()

scheduler = LeagueScheduler(input=input)
scheduler.construction_phase()
scheduler.tabu_phase()

df = scheduler.create_calendar()
scheduler.store_calendar(df, file="out/calendar.xlsx")
```

Type `help(LeagueScheduler)` to show the full documentation.

### Web application

The league scheduler is also made available through a hosted [Streamlit application](https://leaguescheduler.streamlit.app).

It has a more limited set of parameters (namely `m`, `r_max`, `n_iterations`, and `penalties`) but can be used out of the box yet without logging. 

Additionally, the output file includes for every league and by team the distribution of the **number of _adjusted_ rest days between games** (meaning that unavailable dates by that team are not considered in the count of the rest days), as well as the **unused home time slots per team**. This facilitates post-analysis of the quality of the generated calendar.

If the app happens to be sleeping due to inactivity 😴, just wake it up. You can run the app locally with `make web`.

### Timings

How long does the scheduler take? This table sheds some baseline light:

|                  | 4 teams  | 13 teams |
|------------------|--------- | -------- |
| 10 iterations    | ~2s      | ~4s      |
| 100 iterations   | ~3s      | ~5s      |
| 1000 iterations  | ~18s     | ~22s     |
| 10000 iterations | ~146s    | ~186s    |

_Run on a few years old Windows 10 Pro machine with Intel i7–7700HQ CPU and 32GB RAM._

A few 100s iterations are typically sufficient to arrive at a good schedule.

# Feedback?

Let me know if you have any feedback or suggestions for improvement!
