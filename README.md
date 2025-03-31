# Simple Data Cleaner

The Simple Data Cleaner is a Streamlit-based web application designed to clean datasets by handling missing values and outliers automatically.

## Features

1. **Upload Data**: Upload CSV files and preview the dataset.
2. **Automatic Cleaning**:
   - Handle missing values using Mean, Median, or Mode.
   - Handle outliers by capping them within a calculated range.
3. **Download Cleaned Data**: Download the processed dataset as a CSV file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vinay-852/EDA.git
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
2. Upload your dataset in CSV format.
3. View the cleaned dataset and download it as a CSV file.

## Requirements

- Python 3.7 or higher
- Libraries:
  - `streamlit`
  - `pandas`

Install these using the `requirements.txt` file.

## License

This project is licensed under the MIT License.
