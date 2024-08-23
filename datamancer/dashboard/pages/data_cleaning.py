import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from openai import OpenAI
import json

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from datamancer.cleaner import smart_clean_extended
    from datamancer.insights import generate_data_report, plot_data_insights
    from datamancer.type_infer import infer_types, TypeInformation
    DATAMANCER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing from datamancer. {str(e)}")
    DATAMANCER_AVAILABLE = False

# Initialize OpenAI client
client = None

def initialize_openai_client():
    global client
    try:
        api_key = st.secrets["openai_api_key"]
    except FileNotFoundError:
        st.warning("No secrets file found. You'll need to input your OpenAI API key manually.")
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
    
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        st.error("Please enter a valid OpenAI API key to use the AI-powered SQL generation feature.")

def generate_sql_script_ai(df: pd.DataFrame, table_name: str, db_type: str = 'sqlite') -> str:
    """
    Generate a SQL script using AI to create a table and insert data based on the input DataFrame.
    
    :param df: Input DataFrame
    :param table_name: Name of the table to be created
    :param db_type: Type of database (sqlite, mysql, postgresql)
    :return: SQL script as a string
    """
    if not client:
        raise ValueError("OpenAI client is not initialized. Please provide a valid API key.")

    # Prepare the data for the AI model
    column_info = df.dtypes.to_dict()
    column_info = {col: str(dtype) for col, dtype in column_info.items()}
    
    # Sample data (first 5 rows)
    sample_data = df.head().to_dict(orient='records')

    # Prepare the prompt for the AI model
    prompt = f"""
    Generate a SQL script for a {db_type} database to create a table named '{table_name}' and insert data.
    The table should have the following columns:
    {json.dumps(column_info, indent=2)}

    Here's a sample of the data:
    {json.dumps(sample_data, indent=2)}

    Please create a SQL script that:
    1. Creates the table with appropriate data types
    2. Includes any necessary constraints (e.g., primary keys, not null)
    3. Inserts the sample data
    4. Adds any indexes that might be beneficial
    5. Includes any additional optimizations specific to {db_type}

    Only return the SQL script, without any additional explanation.
    """

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a SQL expert assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the SQL script from the API response
    sql_script = response.choices[0].message.content.strip()

    return sql_script

def show():
    st.title("Data Cleaning and AI-Powered SQL Generation")

    initialize_openai_client()

    if 'data' not in st.session_state:
        st.warning("Please upload data first in the 'Upload Data' page.")
        return

    df = st.session_state.data

    st.subheader("Original Data Preview")
    st.write(df.head())

    st.subheader("Data Cleaning Options")
    
    # Cleaning options
    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    handle_missing = st.checkbox("Handle missing values", value=True)
    handle_outliers = st.checkbox("Handle outliers", value=True)
    normalize_data = st.checkbox("Normalize numerical data", value=False)

    # Additional options
    variance_threshold = st.slider("Variance threshold for feature selection", 0.0, 1.0, 0.1, 0.01)
    skew_threshold = st.slider("Skew threshold for logarithmic transformation", 0.0, 2.0, 0.5, 0.1)

    if st.button("Clean Data"):
        with st.spinner("Cleaning data..."):
            if DATAMANCER_AVAILABLE:
                cleaned_df = smart_clean_extended(
                    df,
                    variance_threshold=variance_threshold,
                    skew_threshold=skew_threshold
                )
            else:
                cleaned_df = df.copy()
                st.warning("Advanced cleaning functions are not available. Performing basic cleaning only.")

            if remove_duplicates:
                cleaned_df = cleaned_df.drop_duplicates()

            if handle_missing:
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype in ['int64', 'float64']:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

            if handle_outliers:
                for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    cleaned_df[col] = cleaned_df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

            if normalize_data:
                for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                    cleaned_df[col] = (cleaned_df[col] - cleaned_df[col].min()) / (cleaned_df[col].max() - cleaned_df[col].min())

        st.session_state.cleaned_data = cleaned_df
        st.success("Data cleaned successfully!")

        st.subheader("Cleaned Data Preview")
        st.write(cleaned_df.head())

        st.subheader("Cleaning Summary")
        st.write(f"Original shape: {df.shape}")
        st.write(f"Cleaned shape: {cleaned_df.shape}")

        if DATAMANCER_AVAILABLE:
            # Generate and display data report
            with st.expander("View Data Report"):
                report = generate_data_report(cleaned_df)
                st.json(report)

            # Plot data insights
            with st.expander("View Data Insights"):
                plot_data_insights(cleaned_df)
        else:
            st.warning("Data report and insights are not available due to missing dependencies.")

    st.subheader("AI-Powered SQL Generation")
    table_name = st.text_input("Enter table name for SQL script:", "my_table")
    db_type = st.selectbox("Select database type:", ["sqlite", "mysql", "postgresql"])

    if st.button("Generate SQL Script with AI"):
        if not client:
            st.error("Please enter your OpenAI API key in the text box above.")
        elif 'cleaned_data' in st.session_state:
            with st.spinner("Generating SQL script with AI..."):
                try:
                    sql_script = generate_sql_script_ai(st.session_state.cleaned_data, table_name, db_type)
                    st.text_area("Generated SQL Script:", sql_script, height=300)
                    
                    # Store the SQL script in session state
                    st.session_state.sql_script = sql_script
                    
                    # Option to download the SQL script
                    st.download_button(
                        label="Download SQL Script",
                        data=sql_script,
                        file_name=f"{table_name}_script.sql",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"An error occurred while generating the SQL script: {str(e)}")
        else:
            st.warning("Please clean the data first before generating SQL script.")

    if st.button("Download Cleaned Data"):
        if 'cleaned_data' in st.session_state:
            csv = st.session_state.cleaned_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("Please clean the data first.")

if __name__ == "__main__":
    show()