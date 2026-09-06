"""Load the CSV data into the database."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session

from find_ideal_func.errors import DatavalidatorError

class CSVLoader:
    """Class to load CSV data into the database."""

    table_name: str
    expected_columns: list[str]

    def __init__(self, csv_path: str | Path)-> None:
        """
        Initialize the CSVLoader with the path to the CSV file.

        Args:
            csv_path (str | Path): The path to the CSV file.
        """
        self.csv_path = Path(csv_path)
        self.dataframe: pd.DataFrame | None = None
    
    def read_csv(self) -> pd.DataFrame:
        """Read the CSV file into a pandas DataFrame."""
        try:
            if not self.csv_path.exists():
                raise DatavalidatorError(f"CSV file does not exist: {self.csv_path}")
            self.dataframe = pd.read_csv(self.csv_path)
        except Exception as e:
            raise DatavalidatorError(f"Error reading CSV file: {e}")
        return self.dataframe
    
    def load_df(self,session: Session) -> pd.DataFrame:
        """Reead the CSV and load the data into the database."""
        df = self.read_csv()
        if not all(col in df.columns for col in self.expected_columns):
            raise DatavalidatorError(f"CSV file must contain the following columns: {self.expected_columns}")
        df.to_sql(self.table_name, con=session.get_bind(), if_exists='replace', index=False)
        return df

class IFCSVLoader(CSVLoader):
    """Class to load ideal function CSV data into the database."""

    table_name = "IdealFunction"
    x = ["x"]
    expected_columns = x + [f"y{i}" for i in range(1, 51)]  # y1 to y50

class TestDataCSVLoader(CSVLoader):
    """Class to load raw test data CSV into the database."""

    table_name = "TestData"
    expected_columns = ["x", "y"]

class TrainingDataCSVLoader(CSVLoader):
    """Class to load training data CSV into the database."""

    table_name = "TrainingData"
    expected_columns = ["x", "y1", "y2", "y3", "y4"]