# DLMDSPWP01: Data Processing and Visualization with Python

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Bokeh](https://img.shields.io/badge/Bokeh-2.4.0-green.svg)](https://bokeh.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3.0-orange.svg)](https://pandas.pydata.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-1.4.22-yellow.svg)](https://www.sqlalchemy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project addresses the assignment for the course DLMDSPWP01 (Programming with Python) and provides a comprehensive Python-based solution for processing, analyzing, and visualizing data. The key functionalities include:

- Loading data from CSV files into a SQLite database.
- Finding the best fitting functions for a given set of training data.
- Mapping test data to these ideal functions.
- Visualizing the results using Bokeh.

## Project Structure

```graphql 
DLMDSPWP01/
│
├── data/ # Directory for input data files
│ ├── ideal.csv # Ideal functions data
│ ├── test.csv # Test data
│ └── train.csv # Training data
│
├── main.py # Main script to run the application
│
├── processor.py # Module containing DataHandler and DataProcessor classes
│
├── requirements.txt # List of dependencies
│
├── tests.py # Unit tests for the DataProcessor class
│
└── README.md # This README file

```



## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Ensure you have Python 3.8 or higher installed. You can download it from the [official Python website](https://www.python.org/).

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/DLMDSPWP01.git
    cd DLMDSPWP01
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Place your data files (`ideal.csv`, `test.csv`, and `train.csv`) in the `data/` directory. The structure of each file should match the expectations outlined in the Data Files section below.

### Data Files

- **ideal.csv**: Contains the ideal functions. The file should include an `x` column followed by one or more columns representing different ideal functions.
- **train.csv**: Training data with an `x` column and multiple columns representing different sets of training data.
- **test.csv**: Test data with `x` and `y` columns.

### Usage

To run the main application, simply execute:

```bash
python main.py
```
This script will:

1. Load the CSV files into the SQLite database.
2. Retrieve and process the data to find the best fitting functions.
3. Map the test data to these functions.
4. Save the results to the database and generate a visualization in an HTML file.

## Unit Tests
To run the unit tests for the project:
    
```bash
python tests.py
```
The tests cover:

- Data loading and retrieval.
- Best fit function identification.
- Test data mapping.
- Saving results to the database.

## Features
- **Data Handling**: Load and manage data using pandas and SQLAlchemy.
- **Best Fit Identification**: Determine the most suitable functions for given training data.
- **Data Mapping**: Map test data to ideal functions based on deviation criteria.
- **Visualization**: Generate interactive plots using Bokeh to visualize the relationships between training data, ideal functions, and test points.
- **Custom Exceptions**: Includes user-defined exception handling for robust error management.

## Dependencies
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `bokeh`: For interactive data visualization.
- `SQLAlchemy`: For database interactions.

Check the `requirements.txt` for specific versions.


## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Inspiration from data analysis and visualization projects.
Support from the open-source community.

## Contact
For any inquiries or issues, please contact:

Hana Khoshrou

[Hannaneh.khoshrou@gmail.com](mailto:)

[GitHub Profile]()
