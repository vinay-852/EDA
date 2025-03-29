import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Enhanced EDA App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Data Inspection", "Handle Missing Values", 
                                  "Handle Outliers", "Data Characteristics", "Visualizations"])

# Utility function to download DataFrame as CSV
def download_df(df, filename="processed_data.csv"):
    """
    Utility function to download a DataFrame as a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be downloaded.
    - filename (str): The name of the file to be downloaded.

    Returns:
    - None
    """
    csv = df.to_csv(index=False)
    st.download_button(label="Download Processed Data", data=csv, file_name=filename, mime="text/csv")

# Page: Upload Data
if page == "Upload Data":
    """
    Page for uploading a dataset. Allows users to upload a CSV file and preview the dataset.
    """
    st.title("Upload Your Dataset")
    st.write("Upload a CSV file to begin exploratory data analysis.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.success("File uploaded successfully!")
            st.write("### Dataset Preview")
            st.dataframe(df.head())
            download_df(df, "uploaded_data.csv")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

# Page: Data Inspection
elif page == "Data Inspection":
    """
    Page for inspecting the dataset. Displays dataset preview, shape, column names, data types, 
    unique values, basic statistics, and missing values.
    """
    st.title("Data Inspection")
    if 'df' in st.session_state:
        df = st.session_state['df']
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Dataset Preview")
            st.dataframe(df.head())
            st.write("### Shape of the Dataset")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.write("### Column Names")
            st.write(df.columns.tolist())
        with col2:
            st.write("### Data Types")
            st.write(df.dtypes)
            st.write("### Unique Values per Column")
            st.write(df.nunique())
        st.write("### Basic Statistics")
        st.write(df.describe(include='all'))
        st.write("### Missing Values")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.write(missing_values[missing_values > 0])
        else:
            st.write("No missing values found.")
        download_df(df)
    else:
        st.warning("Please upload a dataset first.")

# Page: Handle Missing Values
elif page == "Handle Missing Values":
    """
    Page for handling missing values in the dataset. Provides various strategies to handle missing values.
    """
    st.title("Handle Missing Values")
    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.write("### Columns with Missing Values")
            st.write(missing_values[missing_values > 0])
            strategy = st.selectbox("Select a strategy to handle missing values:", 
                                    ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Drop Rows", 
                                     "Custom Value", "Automatic", "Do Nothing"], index=7)
            apply_changes = st.button("Apply Changes")
            if apply_changes:
                if strategy == "Automatic":
                    st.write("#### Automatic Strategy Details")
                    for column in df.columns:
                        if df[column].isnull().sum() > 0:
                            if df[column].dtype in ['float64', 'int64']:
                                skewness = abs(df[column].skew())
                                if skewness > 1:
                                    df[column] = df[column].fillna(df[column].median())
                                    st.write(f"'{column}': Filled with median (skewness = {skewness:.2f}).")
                                else:
                                    df[column] = df[column].fillna(df[column].mean())
                                    st.write(f"'{column}': Filled with mean (skewness = {skewness:.2f}).")
                            else:
                                df[column] = df[column].fillna(df[column].mode()[0])
                                st.write(f"'{column}': Filled with mode.")
                elif strategy == "Mean":
                    df = df.fillna(df.mean(numeric_only=True))
                elif strategy == "Median":
                    df = df.fillna(df.median(numeric_only=True))
                elif strategy == "Mode":
                    df = df.fillna(df.mode().iloc[0])
                elif strategy == "Forward Fill":
                    df = df.fillna(method='ffill')
                elif strategy == "Backward Fill":
                    df = df.fillna(method='bfill')
                elif strategy == "Drop Rows":
                    df = df.dropna()
                elif strategy == "Custom Value":
                    custom_value = st.text_input("Enter the custom value:")
                    if custom_value:
                        try:
                            df = df.fillna(float(custom_value) if custom_value.replace('.', '', 1).isdigit() else custom_value)
                        except ValueError:
                            st.error("Invalid custom value.")
                if strategy != "Do Nothing":
                    st.session_state['df'] = df
                    st.write("### Updated Dataset Preview")
                    st.dataframe(df.head())
                    download_df(df)
        else:
            st.success("No missing values found in the dataset.")
    else:
        st.warning("Please upload a dataset first.")

# Page: Handle Outliers
elif page == "Handle Outliers":
    """
    Page for handling outliers in the dataset. Provides various strategies to detect and handle outliers.
    """
    st.title("Handle Outliers")
    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numerical_columns:
            outlier_column = st.selectbox("Select a column to detect and handle outliers:", numerical_columns)
            if outlier_column:
                Q1 = df[outlier_column].quantile(0.25)
                Q3 = df[outlier_column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[outlier_column] < lower_bound) | (df[outlier_column] > upper_bound)]
                st.write(f"### Outliers in '{outlier_column}'")
                st.write(outliers)
                handle_outliers = st.selectbox("Select a strategy to handle outliers:", 
                                               ["Remove Outliers", "Cap Outliers", "Replace with Mean", 
                                                "Replace with Median", "Do Nothing"])
                apply_changes = st.button("Apply Changes")
                if apply_changes:
                    if handle_outliers == "Remove Outliers":
                        df = df[(df[outlier_column] >= lower_bound) & (df[outlier_column] <= upper_bound)]
                        st.write(f"Outliers removed from '{outlier_column}'.")
                    elif handle_outliers == "Cap Outliers":
                        df[outlier_column] = df[outlier_column].clip(lower=lower_bound, upper=upper_bound)
                        st.write(f"Outliers capped to [{lower_bound:.2f}, {upper_bound:.2f}].")
                    elif handle_outliers == "Replace with Mean":
                        mean_value = df[outlier_column].mean()
                        df[outlier_column] = df[outlier_column].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)
                        st.write(f"Outliers replaced with mean: {mean_value:.2f}.")
                    elif handle_outliers == "Replace with Median":
                        median_value = df[outlier_column].median()
                        df[outlier_column] = df[outlier_column].apply(lambda x: median_value if x < lower_bound or x > upper_bound else x)
                        st.write(f"Outliers replaced with median: {median_value:.2f}.")
                    if handle_outliers != "Do Nothing":
                        st.session_state['df'] = df
                        st.write("### Updated Dataset Preview")
                        st.dataframe(df.head())
                        download_df(df)
        else:
            st.info("No numerical columns available for outlier detection.")
    else:
        st.warning("Please upload a dataset first.")

# Page: Data Characteristics
elif page == "Data Characteristics":
    """
    Page for exploring data characteristics. Displays data types, unique values, insights for categorical 
    and numerical columns, and statistical measures.
    """
    st.title("Data Characteristics")
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.write("### Data Types")
        st.write(df.dtypes)
        st.write("### Unique Values per Column")
        st.write(df.nunique())
        st.write("### Categorical Column Insights")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            st.write(f"#### Top 10 Values in '{col}'")
            st.write(df[col].value_counts().head(10))
        st.write("### Numerical Column Insights")
        numerical_df = df.select_dtypes(include=['number'])
        if not numerical_df.empty:
            st.write("#### Correlation Matrix")
            st.write(numerical_df.corr())
            st.write("#### Skewness")
            st.write(numerical_df.skew())
            st.write("#### Kurtosis")
            st.write(numerical_df.kurt())
            st.write("#### Variance")
            st.write(numerical_df.var())
            st.write("#### Standard Deviation")
            st.write(numerical_df.std())
        download_df(df)
    else:
        st.warning("Please upload a dataset first.")

# Page: Visualizations
elif page == "Visualizations":
    """
    Page for creating visualizations. Provides options for numerical, categorical, and mixed types visualizations.
    """
    st.title("Visualizations")
    if 'df' in st.session_state:
        df = st.session_state['df']
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Numerical Columns Visualizations
        st.header("Numerical Columns Visualizations")
        if numerical_cols:
            melted_df = df.melt(value_vars=numerical_cols)
            st.plotly_chart(px.histogram(melted_df, x='value', facet_col='variable', title="Histograms of Numerical Columns", nbins=20))
            st.plotly_chart(px.box(melted_df, y='value', facet_col='variable', title="Box Plots of Numerical Columns"))
            if len(numerical_cols) >= 2:
                corr = df[numerical_cols].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", title="Correlation Heatmap"))
            if 2 <= len(numerical_cols) <= 5:
                st.plotly_chart(px.scatter_matrix(df[numerical_cols], title="Pair Plot"))
        else:
            st.write("No numerical columns found.")

        # Categorical Columns Visualizations
        st.header("Categorical Columns Visualizations")
        if categorical_cols:
            for col in categorical_cols:
                st.plotly_chart(px.bar(df, x=col, title=f"Bar Chart of {col}"))
        else:
            st.write("No categorical columns found.")

        # Mixed Types Visualizations
        st.header("Mixed Types Visualizations")
        if numerical_cols and categorical_cols:
            cat_col = min(categorical_cols, key=lambda col: df[col].nunique())
            for num_col in numerical_cols:
                st.plotly_chart(px.box(df, x=cat_col, y=num_col, title=f"Box Plot of {num_col} by {cat_col}"))
        else:
            st.write("No mixed types visualizations possible (requires both numerical and categorical columns).")
    else:
        st.warning("Please upload a dataset first.")