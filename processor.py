import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from sqlalchemy import create_engine


# Custom Exception for Invalid Data
class InvalidDataError(Exception):
    pass


# Base Class for Data Handling
class DataHandler:
    def __init__(self, db_path='sqlite:///data.db'):
        self.engine = create_engine(db_path)

    def load_csv_to_db(self, file_path, table_name):
        try:
            df = pd.read_csv(file_path)
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        except Exception as e:
            raise InvalidDataError(f"Failed to load data from {file_path}: {e}")


# Derived Class for Specific Data Processing Tasks
class DataProcessor(DataHandler):
    def __init__(self, db_path='sqlite:///data.db'):
        super().__init__(db_path)

    def get_data(self, table_name):
        query = f"SELECT * FROM {table_name}"
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            raise InvalidDataError(f"Failed to retrieve data from {table_name}: {e}")

    def find_best_fit(self, train_df, ideal_df):
        best_fits = {}
        for train_col in train_df.columns[1:]:
            x_train = train_df['x']
            y_train = train_df[train_col]
            min_deviation = float('inf')
            best_fit = None
            for ideal_col in ideal_df.columns[1:]:
                y_ideal = ideal_df[ideal_col]
                deviation = np.sum((y_train - y_ideal) ** 2)
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_fit = ideal_col
            best_fits[train_col] = best_fit
        return best_fits

    def map_test_data(self, test_df, best_fits, train_df, ideal_df):
        results = []
        for _, test_row in test_df.iterrows():
            x_test, y_test = test_row['x'], test_row['y']
            matched = False
            for train_col, ideal_col in best_fits.items():
                max_deviation = np.max(np.abs(train_df[train_col] - ideal_df[ideal_col]))
                ideal_y = ideal_df[ideal_df['x'] == x_test][ideal_col].values[0]
                deviation = np.abs(y_test - ideal_y)
                if deviation <= max_deviation * np.sqrt(2):
                    results.append({
                        'x': x_test,
                        'y': y_test,
                        'ideal_function': ideal_col,
                        'deviation': deviation
                    })
                    matched = True
                    break
            if not matched:
                results.append({
                    'x': x_test,
                    'y': y_test,
                    'ideal_function': None,
                    'deviation': None
                })
        return pd.DataFrame(results)

    def save_results_to_db(self, results_df, table_name):
        try:
            results_df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        except Exception as e:
            raise InvalidDataError(f"Failed to save results to {table_name}: {e}")

    def visualize_data(self, train_df, test_df, ideal_df, best_fits, results_df):
        output_file("visualization.html")
        plots = []
        for train_col, ideal_col in best_fits.items():
            p = figure(title=f"Training Data: {train_col} vs Ideal Function: {ideal_col}", x_axis_label='x',
                       y_axis_label='y')
            p.circle(train_df['x'], train_df[train_col], size=5, color="navy", alpha=0.5,
                     legend_label=f"Train {train_col}")
            p.line(ideal_df['x'], ideal_df[ideal_col], line_width=2, color="green", legend_label=f"Ideal {ideal_col}")
            test_points = results_df[results_df['ideal_function'] == ideal_col]
            p.diamond(test_points['x'], test_points['y'], size=10, color="red", alpha=0.5, legend_label="Test Points")
            plots.append(p)
        grid = gridplot(plots, ncols=2)
        save(grid)