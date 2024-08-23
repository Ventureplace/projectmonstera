import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import json

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
        st.error("Please enter a valid OpenAI API key to use the AI-powered insights feature.")

def generate_ai_insights(df: pd.DataFrame, sql_script: str) -> str:
    """
    Generate insights using AI based on the DataFrame and SQL script.
    
    :param df: Input DataFrame
    :param sql_script: Generated SQL script
    :return: AI-generated insights as a string
    """
    if not client:
        raise ValueError("OpenAI client is not initialized. Please provide a valid API key.")

    # Prepare the data for the AI model
    data_sample = df.head(5).to_dict(orient='records')
    column_info = {col: str(dtype) for col, dtype in df.dtypes.items()}

    prompt = f"""
    Analyze the following dataset and SQL script, and provide insights:

    Data sample:
    {json.dumps(data_sample, indent=2)}

    Column information:
    {json.dumps(column_info, indent=2)}

    SQL Script:
    {sql_script}

    Please provide insights on:
    1. The structure and content of the data
    2. Any patterns or trends you notice
    3. Potential data quality issues
    4. Identify the highest cost drivers based on included fields 
    5. Suggestions for further analysis or visualization

    Provide your analysis in a clear, concise manner.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data analyst providing insights on datasets and SQL scripts."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def show():
    st.title("Data Insights")

    initialize_openai_client()

    if 'data' not in st.session_state or 'sql_script' not in st.session_state:
        st.warning("Please upload data and generate SQL script first.")
        return

    df = st.session_state.data
    sql_script = st.session_state.sql_script

    st.header("Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Data Preview")
        st.write(df.head())

    with col2:
        st.subheader("Generated SQL Script")
        st.code(sql_script, language='sql')

    st.header("Basic Data Statistics")
    st.write(df.describe())

    st.header("Data Types")
    st.write(df.dtypes)

    st.header("Missing Values")
    missing_data = df.isnull().sum()
    st.write(missing_data[missing_data > 0])

    st.header("Correlation Heatmap")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        corr_matrix = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for correlation analysis.")

    st.header("Distribution Plots")
    selected_column = st.selectbox("Select a column for distribution plot:", df.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    if df[selected_column].dtype in ['int64', 'float64']:
        sns.histplot(df[selected_column], kde=True, ax=ax)
    else:
        df[selected_column].value_counts().plot(kind='bar', ax=ax)
    plt.title(f"Distribution of {selected_column}")
    st.pyplot(fig)

    st.header("AI-Generated Insights")
    if st.button("Generate AI Insights"):
        with st.spinner("Generating insights..."):
            try:
                insights = generate_ai_insights(df, sql_script)
                st.markdown(insights)
            except Exception as e:
                st.error(f"An error occurred while generating insights: {str(e)}")

if __name__ == "__main__":
    show()