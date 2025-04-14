import numpy as np
import pandas as pd

from .utils import fill_value


class InputParser:
    """Reads input from Excel file for given league and parses relevant data."""

    def __init__(self, filename: str, unavailable: str = "NIET") -> None:
        """
        Initializes a new instance of the InputParser class.

        Every sheet should be structured as follows:
        - Row 1 has the team names T (starting from column 2).
        - Row 2 has the locations for home games (starting from column 2).
        - Column 1 has the league dates S (starting from row 4) [row 3 is for comments].
        - Cells with value 'unavailable' indicate where a column team is not available (A).
        - Cells with another value are considered the home slot hours, e.g., "20h30" (H).
        - Cells with no value are considered baseline available slots.

        Optionally, a single sheet can be named 'penalties' and contain the penalties
        for the algorithm. The penalties are read as a dictionary with the number of
        days as keys (column 1) and the penalties as values (column 2). In this case,
        the number of days is defined as the exact number of rest days. This class
        internally adds 1 to each key so it conforms to the rest days + 1 convention
        used in SchedulerParams.penalties.

        :param filename: Path location where input Excel file is stored.
        :param unavailable: Cell value to indicate that a team is unavailable.
        """
        self.unavailable = unavailable

        self.file = pd.ExcelFile(filename)
        self.sheet_names = [
            sheet_name
            for sheet_name in self.file.sheet_names
            if sheet_name != "penalties"
        ]

        self.penalties = self.get_penalties()  # same regardless of league sheet

        self.data = None
        self.parsed = False

    def get_penalties(self) -> dict:
        """Reads penalties (if available) from input Excel file and returns them as a dictionary."""
        penalties = None  # fallback

        if "penalties" in self.file.sheet_names:
            penalties = (
                pd.read_excel(self.file, sheet_name="penalties", index_col=0)
                .iloc[:, 0]
                .to_dict()
            )

            # format needs {n_days: penalty} where n_days = rest days + 1 as an int
            # 0 --> e.g., a game on Monday and Tuesday (0 rest days but delta t is 1)
            penalties = {int(k) + 1: v for k, v in penalties.items()}

        return penalties

    def from_excel(self, sheet_name: str = None) -> None:
        """Reads data from input Excel file and sheet, then assigns it to self.data."""
        if sheet_name is None or sheet_name in self.sheet_names:
            data = pd.read_excel(self.file, sheet_name=sheet_name)
            data = data.drop(1, axis=0)  # drop row at index 1 (row 3 in Excel file)
            self.data = data.reset_index(drop=True)
        else:
            raise ValueError(f"Sheet name {sheet_name} not found in file.")

    def parse(self) -> None:
        """Extracts (aka parses) relevant data from input file."""
        data = self.data

        team_names = data.columns[1:]

        # names of locations for home games
        self.locations = {team_name: data[team_name][0] for team_name in team_names}

        # get team indices and names (T)
        teams = dict(enumerate(team_names))

        # get all slots (S)
        dates = pd.to_datetime(data.iloc[1:, 0])  # not necessarily continuous
        slots = dict(enumerate(dates))

        # process core (i.e. without dates and locations) for remaining sets extraction
        self.core = data.iloc[1:, 1:].reset_index(drop=True)

        mat = self.core.copy()
        mat = mat.map(fill_value, unavailable=self.unavailable)
        mat = mat.to_numpy()

        # get all available home slots by team index (H)
        sets_home = {key: np.where(mat[:, key] == 1)[0] for key in teams}

        # get all non-available away slots by team index (A)
        sets_forbidden = {key: np.where(mat[:, key] == -1)[0] for key in teams}

        # assemble sets
        self.sets = {
            "teams": teams,
            "slots": slots,
            "home": sets_home,
            "forbidden": sets_forbidden,
        }

        self.parsed = True
