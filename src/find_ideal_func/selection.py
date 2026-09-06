"""Get the four Ideal functions from the traing data.
For each trainig data point,  chose the idesal function that is minimizing
the sum of the squared y across all shared x values. 
Store the mapping in the database.
"""

from __future__ import annotations
from dataclasses import dataclass
from find_ideal_func.errors import NoMatchingError, IdealFunctionNotFoundError, DatavalidatorError

import pandas as pd
import numpy as np

@dataclass(frozen=True)
class SelectedIdealFunction:
    """Dataclass to hold the selected ideal function for a given training data point."""
    ideal_column: str
    t_column:str
    idf_number:int
    sse: float
    max_delta_y: float

class IdealFunctionSelector:
    """Class to select the ideal function for each training data point."""
    
    def __init__(self, training_data: pd.DataFrame, ideal_functions: pd.DataFrame):
        """
        Initialize the IdealFunctionSelector with training data and ideal functions.

        Args:
            training_data (pd.DataFrame): The training data containing 'x' and 'y' columns.
            ideal_functions (pd.DataFrame): The ideal functions containing 'x' and multiple 'y' columns.
        """
        self.training_data = self.sort_functions(training_data)
        self.ideal_functions = self.sort_functions(ideal_functions)

    def sort_functions(self, functions: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the functions DataFrame by the 'x' column.

        Returns:
            pd.DataFrame: The sorted functions DataFrame.
        """
        return functions.set_index('x').sort_index()

    def _sse(self) -> pd.DataFrame:
        
        idel = self.ideal_functions.to_numpy()
        train = self.training_data.to_numpy()
        diff = train[:, :, None] - idel[:, None, :]
        sse = np.sum(diff**2, axis=0)
        return pd.DataFrame(sse, index=self.training_data.columns, columns=self.ideal_functions.columns)
    
    def select_ideal_function(self) -> list[SelectedIdealFunction]:
        """
        Select the ideal function for each training data point.

        Returns:
            list[SelectedIdealFunction]: A list of SelectedIdealFunction instances for each training data point.
        """
        sse_df = self._sse()
        selected_functions = []

        for t_col in self.training_data.columns:
            row = sse_df.loc[t_col]
            min_sse_col = row.idxmin()
            min_sse_value = row.min()

            # Calculate max delta_y for the selected ideal function
            ideal_y_values = self.ideal_functions[min_sse_col].to_numpy()
            train_y_values = self.training_data[t_col].to_numpy()
            delta_y = np.abs(train_y_values - ideal_y_values)
            max_delta_y = np.max(delta_y)

            idf_number = int(min_sse_col[1:])  # Extract the number from 'yN'

            selected_functions.append(SelectedIdealFunction(
                ideal_column=min_sse_col,
                t_column=t_col,
                idf_number=idf_number,
                sse=min_sse_value,
                max_delta_y=max_delta_y
            ))

        return selected_functions