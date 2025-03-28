import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import io

# Set page configuration for a wider layout and custom title
st.set_page_config(page_title="Enhanced EDA App", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Data Inspection", "Handle Missing Values", "Handle Outliers", 
                                  "Data Characteristics", "Data Visualization", "Automated Report"])

# Utility function to download DataFrame as CSV
def download_df(df, filename="processed_data.csv"):
    csv = df.to_csv(index=False)
    st.download_button(label="Download Processed Data", data=csv, file_name=filename, mime="text/csv")

# Page: Upload Data
if page == "Upload Data":
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
        st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")
        download_df(df)
    else:
        st.warning("Please upload a dataset first.")

# Page: Handle Missing Values
elif page == "Handle Missing Values":
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
                        df = df.fillna(float(custom_value) if custom_value.isnumeric() else custom_value)
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
                st.plotly_chart(px.histogram(df, x=outlier_column, title=f"Distribution of {outlier_column} (Before Handling)"))
                handle_outliers = st.selectbox("Select a strategy to handle outliers:", 
                                               ["Remove Outliers", "Cap Outliers", "Replace with Mean", 
                                                "Replace with Median", "Do Nothing"])
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
                    st.plotly_chart(px.histogram(df, x=outlier_column, title=f"Distribution of {outlier_column} (After Handling)"))
                    download_df(df)
        else:
            st.info("No numerical columns available for outlier detection.")
    else:
        st.warning("Please upload a dataset first.")

# Page: Data Characteristics
elif page == "Data Characteristics":
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

# Page: Data Visualization
elif page == "Data Visualization":
    st.title("Data Visualization")
    if 'df' in st.session_state:
        df = st.session_state['df']
        viz_type = st.selectbox("Select a visualization type:", 
                                ["Scatter Plot", "Histogram", "Box Plot", "Pair Plot", "Heatmap", "Bar Chart", 
                                 "Line Plot", "Violin Plot", "Pie Chart", "Density Plot", "Automatic"])
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if viz_type == "Scatter Plot" and len(numerical_cols) >= 2:
            x_axis = st.selectbox("X-axis:", numerical_cols)
            y_axis = st.selectbox("Y-axis:", numerical_cols, index=1)
            color = st.selectbox("Color (optional):", ["None"] + categorical_cols + numerical_cols)
            st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, color=None if color == "None" else color, 
                                       title=f"Scatter Plot: {x_axis} vs {y_axis}"))
        elif viz_type == "Histogram":
            column = st.selectbox("Column:", df.columns)
            bins = st.slider("Number of bins:", 5, 50, 20)
            st.plotly_chart(px.histogram(df, x=column, nbins=bins, title=f"Histogram of {column}"))
        elif viz_type == "Box Plot" and numerical_cols:
            column = st.selectbox("Column:", numerical_cols)
            st.plotly_chart(px.box(df, y=column, title=f"Box Plot of {column}"))
        elif viz_type == "Pair Plot" and len(numerical_cols) > 1:
            st.plotly_chart(px.scatter_matrix(df, dimensions=numerical_cols, title="Pair Plot"))
        elif viz_type == "Heatmap" and len(numerical_cols) > 1:
            corr = df[numerical_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", title="Correlation Heatmap"))
        elif viz_type == "Bar Chart" and categorical_cols:
            column = st.selectbox("Column:", categorical_cols)
            st.plotly_chart(px.bar(df, x=column, title=f"Bar Chart of {column}"))
        elif viz_type == "Line Plot" and numerical_cols:
            column = st.selectbox("Column:", numerical_cols)
            st.plotly_chart(px.line(df, y=column, title=f"Line Plot of {column}"))
        elif viz_type == "Violin Plot" and numerical_cols:
            column = st.selectbox("Column:", numerical_cols)
            st.plotly_chart(px.violin(df, y=column, box=True, points="all", title=f"Violin Plot of {column}"))
        elif viz_type == "Pie Chart" and categorical_cols:
            column = st.selectbox("Column:", categorical_cols)
            st.plotly_chart(px.pie(df, names=column, title=f"Pie Chart of {column}"))
        elif viz_type == "Density Plot" and numerical_cols:
            column = st.selectbox("Column:", numerical_cols)
            st.plotly_chart(px.density_contour(df, x=column, title=f"Density Plot of {column}"))
        elif viz_type == "Automatic":
            st.write("### Automatic Visualization Dashboard")
            if numerical_cols:
                st.plotly_chart(px.histogram(df, x=numerical_cols[0], title=f"Histogram of {numerical_cols[0]}"))
                st.plotly_chart(px.box(df, y=numerical_cols[0], title=f"Box Plot of {numerical_cols[0]}"))
            if len(numerical_cols) >= 2:
                st.plotly_chart(px.scatter(df, x=numerical_cols[0], y=numerical_cols[1], 
                                          title=f"Scatter Plot: {numerical_cols[0]} vs {numerical_cols[1]}"))
                st.plotly_chart(px.imshow(df[numerical_cols].corr(), text_auto=True, color_continuous_scale="Viridis", 
                                         title="Correlation Heatmap"))
            if categorical_cols:
                st.plotly_chart(px.bar(df, x=categorical_cols[0], title=f"Bar Chart of {categorical_cols[0]}"))
    else:
        st.warning("Please upload a dataset first.")

# Page: Automated Report
elif page == "Automated Report":
    st.title("Automated Report Generation")
    if 'df' in st.session_state:
        df = st.session_state['df']
        if st.button("Generate Report"):
            profile = ProfileReport(df, title="Automated EDA Report", explorative=True)
            st_profile_report(profile)
            report_html = profile.to_html()
            st.download_button(label="Download Report", data=report_html, file_name="eda_report.html", mime="text/html")
    else:
        st.warning("Please upload a dataset first.")