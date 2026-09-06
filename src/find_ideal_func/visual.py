"""Bokeh visualization for the mapping of ideal functions to training data points."""

from __future__ import annotations

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot, column
from bokeh.palettes import Category10

from find_ideal_func.selection import SelectedIdealFunction
from find_ideal_func.mapping import MappingResult

import pandas as pd
import numpy as np

class Vizualizer:
    """Class to visualize the mapping of ideal functions to training data points using Bokeh."""
    
    def __init__(self, t_data: pd.DataFrame, ideal_functions: pd.DataFrame, selected_functions: list[SelectedIdealFunction], mapping_results: list[MappingResult]):
        """
        Initialize the Vizualizer with training data, ideal functions, selected functions, and mapping results.

        Args:
            t_data (pd.DataFrame): The training data containing 'x' and 'y' columns.
            ideal_functions (pd.DataFrame): The ideal functions containing 'x' and multiple 'y' columns.
            selected_functions (list[SelectedIdealFunction]): A list of SelectedIdealFunction instances for each training data point.
            mapping_results (list[MappingResult]): A list of MappingResult instances for each training data point.
        """
        self.t_data = t_data
        self.ideal_functions = ideal_functions
        self.selected_functions = selected_functions
        self.mapping_results = mapping_results

    def render(self, output_html: str) -> None:
        """
        Render the visualization and save it to an HTML file.

        Args:
            output_html (str): The path to the output HTML file.
        """
        output_file(output_html)
        
        # Create a figure for the training data
        p1 = figure(title="Training Data", x_axis_label='x', y_axis_label='y', width=800, height=400)
        p1.scatter(self.t_data['x'], self.t_data['y'], color='blue', legend_label='Training Data', size=5)
        
        # Create a figure for the ideal functions
        p2 = figure(title="Ideal Functions", x_axis_label='x', y_axis_label='y', width=800, height=400)
        colors = Category10[10]
        for i, col in enumerate(self.ideal_functions.columns[1:]):  # Skip 'x' column
            p2.line(self.ideal_functions['x'], self.ideal_functions[col], color=colors[i % len(colors)], legend_label=col)
        
        # Create a figure for the mapping results
        p3 = figure(title="Mapping Results", x_axis_label='x', y_axis_label='y', width=800, height=400)
        for result in self.mapping_results:
            if result.delta_y is not None:
                p3.circle(result.x, result.y, color='red', size=5, legend_label=f'Ideal Function {result.ideal_function_number}')
        
        # Arrange the plots in a grid layout
        grid = gridplot([[p1], [p2], [p3]])
        
        # Save the visualization to an HTML file
        save(grid)