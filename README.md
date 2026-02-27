# ⚽📅 2RR League Scheduler

[![PyPI](https://img.shields.io/pypi/v/leaguescheduler)](https://pypi.org/project/leaguescheduler/)
[![Python](https://img.shields.io/pypi/pyversions/leaguescheduler)](https://pypi.org/project/leaguescheduler/)
[![CI](https://github.com/sborms/leaguescheduler/actions/workflows/ci.yaml/badge.svg)](https://github.com/sborms/leaguescheduler/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leaguescheduler.streamlit.app)

If you are looking to schedule a sports league with at least the following constraints...

- Everyone plays 1 home game and 1 away game against each other (double round-robin)
- Home games are played on reserved dates
- Away games are not played on unavailable dates
- No team plays 2 games on the same day
- The calendar is spread out such that teams have enough rest days between games

... then this will help you!

This software implements **constrained time-relaxed double round-robin (2RR) sports league scheduling** using the tabu search based heuristic algorithm described in the paper [**Scheduling a non-professional indoor football league**](https://pure.tue.nl/ws/portalfiles/portal/121797609/Bulck2019_Article_SchedulingANon_professionalInd.pdf) by Van Bulck, Goosens and Spieksma (2019). The meta-algorithm heavily relies on the Hungarian algorithm to recurrently solve the transportation problem. Some additional tricks were added, especially to minimize excessive rest days (internally fixed at 28) between consecutive games of teams.

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

For instance, a league could consist of 10 teams, each with about 12 to 20 reserved dates and a number of unavailable dates.

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

Alternatively, you can execute `make example` which runs the above example.

See `2rr --help` (and the research paper mentioned at the top) for more information about all the available arguments.

<details>
<summary><b>CLI reference</b></summary>

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-file` | text | *required* | Input Excel file with for every team their (un)availability data |
| `--output-folder` | text | *required* | Folder where the outputs (logs, overview, schedules) will be stored |
| `--seed` | integer | `None` | Optional seed for `np.random.seed()` |
| `--unavailable` | text | `NIET` | Cell value to indicate that a team is unavailable |
| `--clip-bot` | integer | `1` | Value for clipping rest days plot on low end |
| `--clip-upp` | integer | `41` | Value for clipping rest days plot on high end |
| `--net / --no-net` | flag | `--no-net` | Report the adjusted number of rest days |
| `--tabu-length` | integer | `4` | Number of iterations during which a team cannot be selected |
| `--perturbation-length` | integer | `50` | Check perturbation need every this many iterations |
| `--n-iterations` | integer | `10000` | Number of tabu phase iterations |
| `--m` | integer | `7` | Minimum number of time slots between 2 games with same pair of teams |
| `--p` | integer | `1000` | Cost from dummy supply node q to non-dummy demand node |
| `--r-max` | integer | `4` | Minimum required time slots for 2 games of same team |
| `--alpha` | float | `0.5` | Probability of picking perturbation operator 1 |
| `--beta` | float | `0.01` | Probability of removing a game in operator 1 |
| `--cost-excessive-rest-days` | float | `500` | Cost for excessive rest days |

</details>

#### Classes

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

<details>
<summary><b>Python API reference</b></summary>

**`InputParser(filename, unavailable="NIET")`** — Reads and parses input Excel file.

**`SchedulerParams(...)`** — Configuration dataclass for the scheduler.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tabu_length` | int | `4` | Iterations during which a team cannot be selected |
| `perturbation_length` | int | `50` | Check perturbation need every this many iterations |
| `n_iterations` | int | `10000` | Number of tabu phase iterations |
| `m` | int | `7` | Min time slots between 2 games with same pair |
| `p` | int | `1000` | Cost from dummy supply node to non-dummy demand node |
| `r_max` | int | `4` | Min required time slots for 2 games of same team |
| `penalties` | dict | `{}` | Custom penalty mapping for rest days |
| `alpha` | float | `0.5` | Probability of picking perturbation operator 1 |
| `beta` | float | `0.01` | Probability of removing a game in operator 1 |
| `cost_excessive_rest_days` | float | `500` | Cost for excessive rest days |

**`LeagueScheduler(input, params=SchedulerParams(), logger=None)`** — Main scheduler class.

</details>

#### Web application

The league scheduler is also made available through a hosted [Streamlit application](https://leaguescheduler.streamlit.app).

It has a more limited set of parameters (namely `m`, `r_max`, `n_iterations`, and `penalties`) but can be used out of the box yet without logging.

Additionally, the output file includes for every league and by team the distribution of the **number of *adjusted* rest days between games** (meaning that unavailable dates by that team are not considered in the count of the rest days), as well as the **unused home time slots per team**. This facilitates post-analysis of the quality of the generated calendar.

If the app sleeps due to inactivity 😴, just wake it back up. You can run the app locally with `make web`.

#### Timings

How long does the scheduler take? This table sheds some baseline light for a league of **13 teams**:

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

Quite fast. Thanks to Ra-Ra-Rust! 🦀

## Feedback?

Let me know if you have any feedback or suggestions for improvement!
