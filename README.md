# Enhanced EDA App

The Enhanced EDA App is a Streamlit-based web application designed to simplify exploratory data analysis (EDA). It provides tools for inspecting, cleaning, and visualizing datasets in an interactive and user-friendly interface.

## Features

1. **Upload Data**: Upload CSV files and preview the dataset.
2. **Data Inspection**: View dataset structure, column details, basic statistics, and missing values.
3. **Handle Missing Values**: Choose from strategies like Mean, Median, Mode, Forward Fill, Backward Fill, Drop Rows, Custom Value, or Automatic handling.
4. **Handle Outliers**: Detect and handle outliers using strategies like Remove, Cap, Replace with Mean/Median.
5. **Data Characteristics**: Explore data types, unique values, correlations, skewness, and other statistical insights.
6. **Visualizations**: Generate histograms, box plots, correlation heatmaps, bar charts, and more.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd EDA
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

1. Launch the app using the command above.
2. Use the sidebar to navigate between pages:
   - **Upload Data**: Upload your dataset in CSV format.
   - **Data Inspection**: Inspect dataset structure and statistics.
   - **Handle Missing Values**: Handle missing data using various strategies.
   - **Handle Outliers**: Detect and handle outliers in numerical columns.
   - **Data Characteristics**: Explore detailed insights into your data.
   - **Visualizations**: Generate visualizations for better data understanding.

## Requirements

- Python 3.7 or higher
- Libraries:
  - `streamlit`
  - `pandas`
  - `plotly`
  - `dataprep`

Install these using the `requirements.txt` file.

## Contributing

Contributions are welcome! Fork the repository, create a branch, and submit a pull request.

## License

This project is licensed under the MIT License.