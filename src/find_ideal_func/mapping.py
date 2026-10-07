""" Map the selected ideal functions to the training data points. """

from __future__ import annotations

import pandas as pd
import math

from dataclasses import dataclass
from sqlalchemy.orm import Session

from find_ideal_func.database import Mappings
from find_ideal_func.selection import SelectedIdealFunction

@dataclass(frozen=True)
class MappingResult:
    """Dataclass to hold the mapping result for a given training data point."""
    delta_y: float | None
    x: float
    y: float
    ideal_function_number: int | None
    

class TestDataMapper:
    """Class to map the selected ideal functions to the training data points."""
    
    def __init__(self, t_data: pd.DataFrame, ideal_functions: pd.DataFrame, select_functions: list[SelectedIdealFunction], session: Session) -> None:
        """
        Initialize the TestDataMapper with training data, ideal functions, selected functions, and a database session.

        Args:
            t_data (pd.DataFrame): The test data containing 'x' and 'y' columns.
            ideal_functions (pd.DataFrame): The ideal functions containing 'x' and multiple 'y' columns.
            select_functions (list[SelectedIdealFunction]): A list of SelectedIdealFunction instances for each training data point.
            session (Session): The SQLAlchemy session for database operations.
        """
        self.t_data = t_data
        self.ideal_functions = ideal_functions.set_index("x").sort_index()
        self.select_functions = select_functions
        self.session = session
        

    def map_data(self) -> list[MappingResult]:
      
    
        def _threshold() -> dict[str, tuple[int, float]]:
            """Map ideal column for each ideal function to the training data point."""
            return {sf.ideal_column: (sf.idf_number,sf.max_delta_y * math.sqrt(2)) for sf in self.select_functions}
        
        results = []
        treshold_dict = _threshold()

        for _, row in self.t_data.iterrows():
            x, y = row['x'], row['y']

            if x not in self.ideal_functions.index:
                results.append(MappingResult(delta_y=None, x=x, y=y, ideal_function_number=None))
                continue
            
            idf_number: int | None = None
            delta_y: float | None = None

            for ideal_column, (idf_num, allowed_delta_y) in treshold_dict.items():
                ideal_y = self.ideal_functions.at[x, ideal_column]
                current_delta_y = abs(y - ideal_y)

                if current_delta_y <= allowed_delta_y:
                    if delta_y is None or current_delta_y < delta_y:
                        idf_number = idf_num
                        delta_y = current_delta_y
            results.append(MappingResult(delta_y=delta_y, x=x, y=y, ideal_function_number=idf_number))

        return results
    
    def save_results(self, results: list[MappingResult]) -> None:
        """Save the successfully mapped test points to the database."""
        for result in results:
            if result.ideal_function_number is None:
                continue
            mapping = Mappings(
                x=result.x,
                y=result.y,
                delta_y=result.delta_y,
                ideal_function_number=result.ideal_function_number
            )
            self.session.add(mapping)
        self.session.commit()
