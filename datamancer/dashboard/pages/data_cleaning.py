import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from openai import OpenAI
import json
from io import BytesIO

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

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

    # Sanitize table name
    table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
    
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

    if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
        st.warning("Please upload data files first in the 'Upload Data' page.")
        return

    # File selection
    file_names = [file['name'] for file in st.session_state.uploaded_files]
    selected_files = st.multiselect("Select files to clean:", file_names, default=[file_names[0]])

    if not selected_files:
        st.warning("Please select at least one file to clean.")
        return

    # Cleaning options
    st.subheader("Data Cleaning Options")
    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    handle_missing = st.checkbox("Handle missing values", value=True)
    handle_outliers = st.checkbox("Handle outliers", value=True)
    normalize_data = st.checkbox("Normalize numerical data", value=False)

    # Additional options
    variance_threshold = st.slider("Variance threshold for feature selection", 0.0, 1.0, 0.1, 0.01)
    skew_threshold = st.slider("Skew threshold for logarithmic transformation", 0.0, 2.0, 0.5, 0.1)

    if st.button("Clean Selected Data"):
        cleaned_dfs = {}
        for file_name in selected_files:
            file_data = next(file for file in st.session_state.uploaded_files if file['name'] == file_name)
            df = pd.read_csv(BytesIO(file_data['content'])) if file_data['type'] == '.csv' else pd.read_excel(BytesIO(file_data['content']))
            
            st.subheader(f"Cleaning: {file_name}")
            with st.spinner(f"Cleaning {file_name}..."):
                cleaned_df = clean_dataframe(df, remove_duplicates, handle_missing, handle_outliers, normalize_data, variance_threshold, skew_threshold)
            
            cleaned_dfs[file_name] = cleaned_df
            st.success(f"{file_name} cleaned successfully!")

            st.subheader(f"Cleaned Data Preview: {file_name}")
            st.write(cleaned_df.head())

            st.subheader(f"Cleaning Summary: {file_name}")
            st.write(f"Original shape: {df.shape}")
            st.write(f"Cleaned shape: {cleaned_df.shape}")

            if DATAMANCER_AVAILABLE:
                with st.expander(f"View Data Report: {file_name}"):
                    report = generate_data_report(cleaned_df)
                    st.json(report)

                with st.expander(f"View Data Insights: {file_name}"):
                    plot_data_insights(cleaned_df)

        # Store cleaned data in session state
        st.session_state.cleaned_data = cleaned_dfs
        st.success("All selected files have been cleaned and stored!")

    # AI-Powered SQL Generation
    st.subheader("AI-Powered SQL Generation")
    db_type = st.selectbox("Select database type:", ["sqlite", "mysql", "postgresql"])

    if st.button("Generate SQL Scripts with AI"):
        if not client:
            st.error("Please enter your OpenAI API key in the text box above.")
        elif 'cleaned_data' in st.session_state and st.session_state.cleaned_data:
            sql_scripts = {}
            for file_name, cleaned_df in st.session_state.cleaned_data.items():
                table_name = file_name.split('.')[0]  # Use filename without extension as table name
                with st.spinner(f"Generating SQL script for {file_name}..."):
                    try:
                        sql_script = generate_sql_script_ai(cleaned_df, table_name, db_type)
                        sql_scripts[file_name] = sql_script
                        
                        st.subheader(f"SQL Script for {file_name}")
                        st.text_area(f"Generated SQL Script for {file_name}:", sql_script, height=200)
                        
                        # Option to download individual SQL script
                        st.download_button(
                            label=f"Download SQL Script for {file_name}",
                            data=sql_script,
                            file_name=f"{table_name}_script.sql",
                            mime="text/plain",
                        )
                    except Exception as e:
                        st.error(f"An error occurred while generating the SQL script for {file_name}: {str(e)}")
            
            # Store all SQL scripts in session state
            st.session_state.sql_scripts = sql_scripts
            
            # Option to download all SQL scripts as a single file
            if len(sql_scripts) > 1:
                all_scripts = "\n\n-- Next Table --\n\n".join(sql_scripts.values())
                st.download_button(
                    label="Download All SQL Scripts",
                    data=all_scripts,
                    file_name="all_tables_script.sql",
                    mime="text/plain",
                )
        else:
            st.warning("Please clean the data first before generating SQL scripts.")

    # Download options for cleaned data
    if 'cleaned_data' in st.session_state and st.session_state.cleaned_data:
        st.subheader("Download Cleaned Data")
        for file_name, cleaned_df in st.session_state.cleaned_data.items():
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                label=f"Download Cleaned {file_name}",
                data=csv,
                file_name=f"cleaned_{file_name}",
                mime="text/csv",
            )

# Add this function to handle the cleaning process
def clean_dataframe(df, remove_duplicates, handle_missing, handle_outliers, normalize_data, variance_threshold, skew_threshold):
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

    return cleaned_df

if __name__ == "__main__":
    show()