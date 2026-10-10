"""Unit tests for find_ideal_func.visual.

Bokeh output is only meaningfully checked visually in a browser, so these
tests just verify that rendering completes without error and produces a
non-empty HTML file, per the "where possible" unit-test criterion.
"""

from __future__ import annotations

import pandas as pd

from find_ideal_func.mapping import MappingResult
from find_ideal_func.selection import SelectedIdealFunction
from find_ideal_func.visual import Vizualizer


def test_render_writes_html_file(tmp_path):
    training_data = pd.DataFrame({"x": [0.0, 1.0], "y1": [0.0, 1.0]})
    test_data = pd.DataFrame({"x": [0.0, 1.0], "y": [0.1, 0.9]})
    ideal_functions = pd.DataFrame({"x": [0.0, 1.0], "y1": [0.0, 1.0]})
    selected_functions = [
        SelectedIdealFunction(
            ideal_column="y1", t_column="y1", idf_number=1, sse=0.01, max_delta_y=0.1
        )
    ]
    mapping_results = [
        MappingResult(delta_y=0.1, x=0.0, y=0.1, ideal_function_number=1),
        MappingResult(delta_y=None, x=1.0, y=0.9, ideal_function_number=None),
    ]

    out_file = tmp_path / "viz.html"
    Vizualizer(
        training_data, test_data, ideal_functions, selected_functions, mapping_results
    ).render(str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 0
