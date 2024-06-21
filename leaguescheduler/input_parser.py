import numpy as np
import pandas as pd

from .utils import fill_value


class InputParser:
    """Reads input from Excel file for given league and extracts relevant data."""

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

        :param filename: Path location where input Excel file is stored.
        :param unavailable: Cell value to indicate that a team is unavailable.
        """
        self.unavailable = unavailable

        self.file = pd.ExcelFile(filename)
        self.sheet_names = self.file.sheet_names

        self.data = None

    def read(self, sheet_name: str = None) -> None:
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
        teams = {key: team_name for key, team_name in enumerate(team_names, start=0)}

        # get all slots (S)
        dates = pd.to_datetime(data.iloc[1:, 0])  # not necessarily continuous
        slots = {key: date for key, date in enumerate(dates, start=0)}

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
