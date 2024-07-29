# 2RR League Scheduler

![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leaguescheduler.streamlit.app/)

This repository implements **constrained time-relaxed double round-robin (2RR) sports league scheduling** using the tabu search based heuristic algorithm described in the paper [**Scheduling a non-professional indoor football league**](https://pure.tue.nl/ws/portalfiles/portal/121797609/Bulck2019_Article_SchedulingANon_professionalInd.pdf) by Van Bulck, Goosens and Spieksma (2019). The meta-algorithm heavily relies on the Hungarian algorithm as implemented in the [`munkres` package](https://software.clapper.org/munkres), to solve the transportation problem recurrently.

If you are looking to schedule a sports league with at least the following constraints:
- Everyone plays 1 home game and 1 away game against each other
- Home games are played on reserved dates
- Away games are not played on unavailable dates
- No team plays 2 games on the same day
- The calendar is spread out such that teams have enough rest days between games

... then this will help you!

# Usage

## Input

The Excel file `example_input.xlsx` contains an example of the input data. The input should always, in the exact format as in the example, include the reserved dates (together with location and time) and unavailable dates of all teams from a single league.

One sheet corresponds to one league. 

For instance, a league could consist of 12 teams, each with about 12 to 20 reserved dates and a number of unavailable dates.

## Output

The generated output is an Excel file with one league per sheet and the respective optimal calendar.

The calendar includes the date, time, location, home team and away team for each game. It also includes the unplanned games (at the bottom).

## Scheduling

### CLI

After having cloned the repository you can install the package in editable mode by running `pip install -e .` in the root directory.

Once installed, you can use the scheduler from the command line as follows:

```bash
2rr \
--input_file "example_input.xlsx" \
--output_folder "example_output" \
--seed 505 \
--n_iterations 100
```

Alternatively, you can execute `make 2rr` which runs the above example.

See `2rr --help` (and the research paper mentioned at the top) for more information about all the available arguments. You can also specify a `.json` configuration file and use that as (only) CLI input.

To more freely play around, you can also import the core classes in your own Python script or notebook:

```python
from leaguescheduler import InputParser, LeagueScheduler
```

Type `help(LeagueScheduler)` to show the documentation.

### Web application

The league scheduler is also made available through a [Streamlit application](https://leaguescheduler.streamlit.app/). It has a more limited set of parameters (namely `n_iterations`, `m`, `R_max` and `penalties`) but can be used out of the box yet without logging. The output file includes for every league and by team the distribution of the number of rest days between games, as well as the unused home time slots per team.

Running through 1000 iterations can take up to 1-2 minutes for a league with approx. 13 teams.

If the app happens to be sleeping due to inactivity, feel free to wake it back up.

# Feedback?

Let me know if you have any feedback or suggestions for improvement!
