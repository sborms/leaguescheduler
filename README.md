# ⚽📅 2RR League Scheduler

[![PyPI](https://img.shields.io/pypi/v/leaguescheduler)](https://pypi.org/project/leaguescheduler)
[![Python](https://img.shields.io/pypi/pyversions/leaguescheduler)](https://pypi.org/project/leaguescheduler)
[![CI](https://github.com/sborms/leaguescheduler/actions/workflows/ci.yaml/badge.svg)](https://github.com/sborms/leaguescheduler/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leaguescheduler.streamlit.app)

If you are looking to schedule a sports league with at least the following constraints...

- Everyone plays 1 home game and 1 away game against each other (double round-robin)
- Home games are played on reserved dates
- Away games are not played on unavailable dates
- No team plays 2 games on the same day
- The calendar is spread out such that teams have enough rest days between games

... then the `leaguescheduler` Python package will help you!

## Installation

Install from PyPI:

```bash
pip install leaguescheduler
```

Or with [uv](https://docs.astral.sh/uv):

```bash
uv add leaguescheduler
```

For development, clone the repository and run:

```bash
uv venv
uv sync
```

## Usage

### Input

The Excel file `example/input.xlsx` contains an example of the input data. The input should always, in the exact format as in the example, include the reserved dates (together with location and time) and unavailable dates of all teams from a single league.

One sheet corresponds to one league.

A typical league could consist of 10 teams, each with around 15 reserved dates and a number of unavailable dates (e.g. during holidays).

### Output

The generated output for a single league is a solutions matrix `X` that can easily be converted into a clear `DataFrame` calendar, and stored as an Excel file.

The calendar includes the date, time, location, home team and away team for each game. The unplanned games are put at the bottom.

### Scheduling

#### CLI

You can use the scheduler from the command line as follows:

```bash
2rr \
--input_file "example/input.xlsx" \
--output_folder "example/output" \
--seed 321 \
--n_iterations 500
```

The above example can also be run with `make example`.

See `2rr --help` (and the research paper mentioned at the top) for more information about all the available arguments.

#### Classes

To more freely play around, you can import the core classes in your own Python script or notebook.

Here's a minimal example with default parameters:

```python
from leaguescheduler import InputParser, LeagueScheduler

# load input data with teams and their reserved/unavailable dates
input = InputParser(input_file)
input.from_excel(sheet_name=input.sheet_names[0])
input.parse()

# run the optimizer to find a good schedule
scheduler = LeagueScheduler(input=input)
scheduler.construction_phase()
scheduler.tabu_phase()

# convert the schedule into a usable Excel file
df = scheduler.create_calendar()
scheduler.store_calendar(df, file="out/calendar.xlsx")
```

Type `help(LeagueScheduler)` to show the full documentation.

#### Web application

The league scheduler is also made available through a hosted [Streamlit application](https://leaguescheduler.streamlit.app).

It has a more limited set of parameters (namely `m`, `r_max`, `n_iterations`, and `penalties`) but can be used out of the box yet without logging.

Additionally, the output file includes for every league and by team the distribution of the **number of *adjusted* rest days between games** (meaning that unavailable dates by that team are not considered in the count of the rest days), as well as the **unused home time slots per team**. This facilitates post-analysis of the quality of the generated calendar.

If the app sleeps due to inactivity 😴, just wake it back up. You can run the app locally with `make web`.

## Algorithm

The optimizer implements **constrained time-relaxed double round-robin (2RR) sports league scheduling** using the tabu search based heuristic algorithm described in the paper [**Scheduling a non-professional indoor football league**](https://pure.tue.nl/ws/portalfiles/portal/121797609/Bulck2019_Article_SchedulingANon_professionalInd.pdf) by Van Bulck, Goosens and Spieksma (2019).

The used meta-algorithm heavily relies on the Hungarian algorithm to recurrently solve the transportation problem. We include some additional internal tricks, especially to minimize excessive rest days (fixed at 28) between consecutive games of the same team.

### Timings

How long does the scheduler take? This table sheds some baseline light for a league of **13 teams** (156 games):

| Iterations       | Time       |
|------------------|----------- |
| 10               | <1s        |
| 100              | <1s        |
| 1k               | <1s        |
| 10k              | ~3s        |
| 100k             | ~25s       |
| 1M               | ~245s      |

*Run on a few years old Windows 10 Pro machine with Intel i7–7700HQ CPU and 32GB RAM.*

A few 100(0)s iterations are typically sufficient to arrive at a good schedule.

Quite fast, thanks to Ra-Ra-Rust via this own repo's custom Transportation Problem Solver [fasttps](https://pypi.org/project/fasttps/)! 🦀

## Feedback?

Let me know if you have any feedback or suggestions for improvement!
