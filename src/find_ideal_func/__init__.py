"""find_ideal_func: choose the four best-fit ideal functions for the training
data, map the test data onto them, persist everything to SQLite, and render
a Bokeh visualization of the result.
"""

from __future__ import annotations

from pathlib import Path

from find_ideal_func.database import DbManager
from find_ideal_func.errors import DatavalidatorError, IdealFunctionNotFoundError
from find_ideal_func.loaders import IFCSVLoader, TestDataCSVLoader, TrainingDataCSVLoader
from find_ideal_func.mapping import TestDataMapper
from find_ideal_func.selection import IdealFunctionSelector
from find_ideal_func.visual import Vizualizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"


def main() -> None:
    """Run the full ideal-function selection and test-data mapping pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    db_manager = DbManager(f"sqlite:///{OUTPUT_DIR / 'ideal_functions.db'}")
    db_manager.init_db()
    session = db_manager.get_session()

    try:
        training_data = TrainingDataCSVLoader(DATA_DIR / "train.csv").load_df(session)
        ideal_functions = IFCSVLoader(DATA_DIR / "ideal.csv").load_df(session)
        test_data = TestDataCSVLoader(DATA_DIR / "test.csv").load_df(session)
        session.commit()

        selector = IdealFunctionSelector(training_data, ideal_functions)
        selected_functions = selector.select_ideal_function()

        mapper = TestDataMapper(test_data, ideal_functions, selected_functions, session)
        mapping_results = mapper.map_data()
        mapper.save_results(mapping_results)

        visualizer = Vizualizer(training_data, test_data, ideal_functions, selected_functions, mapping_results)
        visualizer.render(str(OUTPUT_DIR / "visualization.html"))

        mapped_count = sum(1 for r in mapping_results if r.ideal_function_number is not None)
        print(f"Selected ideal functions: {[(sf.t_column, sf.ideal_column) for sf in selected_functions]}")
        print(f"Mapped {mapped_count}/{len(mapping_results)} test points.")
        print(f"Database written to: {OUTPUT_DIR / 'ideal_functions.db'}")
        print(f"Visualization written to: {OUTPUT_DIR / 'visualization.html'}")
    except (DatavalidatorError, IdealFunctionNotFoundError) as e:
        print(f"Pipeline failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
