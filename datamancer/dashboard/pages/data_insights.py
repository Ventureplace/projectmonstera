import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json
from io import BytesIO
from typing import List

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

def generate_ai_insights(dfs: List[pd.DataFrame], file_names: List[str]) -> str:
    if not client:
        raise ValueError("OpenAI client is not initialized. Please provide a valid API key.")

    # Prepare the data for the AI model
    data_samples = [df.head(5).to_dict(orient='records') for df in dfs]
    column_infos = [{col: str(dtype) for col, dtype in df.dtypes.items()} for df in dfs]

    prompt = f"""
    Analyze the following datasets and provide insights:

    {', '.join([f"Dataset {i+1} ({file_names[i]}):" for i in range(len(dfs))])}

    {json.dumps([{"name": file_names[i], "sample": data_samples[i], "columns": column_infos[i]} for i in range(len(dfs))], indent=2)}

    Please provide insights on:
    1. The structure and content of each dataset
    2. Any patterns or trends you notice within each dataset
    3. Comparisons between the datasets (if multiple datasets are provided)
    4. Potential data quality issues in each dataset
    5. Suggestions for further analysis or visualization

    If multiple datasets are provided, focus on comparing and contrasting them.
    Provide your analysis in a clear, concise manner.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst providing insights on datasets."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def generate_ai_response(df: pd.DataFrame, user_query: str) -> str:
    """
    Generate AI response based on the DataFrame and user query.
    
    :param df: Input DataFrame
    :param user_query: User's question about the data
    :return: AI-generated response as a string
    """
    if not client:
        raise ValueError("OpenAI client is not initialized. Please provide a valid API key.")

    # Prepare the data for the AI model
    data_sample = df.head(5).to_dict(orient='records')
    column_info = {col: str(dtype) for col, dtype in df.dtypes.items()}

    prompt = f"""
    Analyze the following dataset and answer the user's question:

    Data sample:
    {json.dumps(data_sample, indent=2)}

    Column information:
    {json.dumps(column_info, indent=2)}

    User question: {user_query}

    Please provide a clear and concise answer based on the data.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst providing insights on datasets."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def show_data_summary(df):
    st.subheader("Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Rows", df.shape[0])
    with col2:
        st.metric("Number of Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    with st.expander("Column Types"):
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

def show_quick_insights(df):
    st.subheader("Quick Insights")

    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        with st.expander("Numeric Columns Summary"):
            st.dataframe(df[numeric_cols].describe())

    # Categorical columns summary
    cat_cols = df.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        with st.expander("Categorical Columns Summary"):
            for col in cat_cols:
                st.write(f"{col}: {df[col].nunique()} unique values")
                if df[col].nunique() <= 10:
                    fig = px.pie(df, names=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        with st.expander("Missing Values"):
            st.dataframe(missing[missing > 0])

def plot_interactive_viz(dfs, file_names):
    st.subheader("Interactive Visualizations")

    all_numeric_cols = set()
    for df in dfs:
        all_numeric_cols.update(df.select_dtypes(include=[np.number]).columns)

    if all_numeric_cols:
        selected_cols = st.multiselect("Select numeric columns for visualization:", list(all_numeric_cols), default=[list(all_numeric_cols)[0]])

        if selected_cols:
            viz_type = st.radio("Select visualization type:", ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap"])

            if viz_type == "Histogram":
                for i, df in enumerate(dfs):
                    fig = px.histogram(df, x=selected_cols[0], title=f"Histogram: {file_names[i]}")
                    st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Box Plot":
                fig = go.Figure()
                for i, df in enumerate(dfs):
                    for col in selected_cols:
                        fig.add_trace(go.Box(y=df[col], name=f"{file_names[i]} - {col}"))
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Scatter Plot" and len(selected_cols) >= 2:
                for i, df in enumerate(dfs):
                    fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1], title=f"Scatter Plot: {file_names[i]}")
                    st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Correlation Heatmap":
                for i, df in enumerate(dfs):
                    corr = df[selected_cols].corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto", title=f"Correlation Heatmap: {file_names[i]}")
                    st.plotly_chart(fig, use_container_width=True)

def show_comparative_insights(dfs, file_names):
    if len(dfs) < 2:
        st.warning("Select at least two files to compare.")
        return

    # Compare basic statistics
    st.subheader("Basic Statistics Comparison")
    stats_dfs = [df.describe() for df in dfs]
    stats_df = pd.concat(stats_dfs, axis=1, keys=file_names)
    st.dataframe(stats_df)

    # Compare column presence
    st.subheader("Column Presence Comparison")
    all_columns = set().union(*[df.columns for df in dfs])
    column_presence = pd.DataFrame({name: [col in df.columns for col in all_columns] for name, df in zip(file_names, dfs)}, index=all_columns)
    st.dataframe(column_presence)

    # Compare data types
    st.subheader("Data Types Comparison")
    dtype_dfs = [df.dtypes.rename(name) for df, name in zip(dfs, file_names)]
    dtype_df = pd.concat(dtype_dfs, axis=1)
    st.dataframe(dtype_df)

    # Compare missing values
    st.subheader("Missing Values Comparison")
    missing_dfs = [df.isnull().sum().rename(name) for df, name in zip(dfs, file_names)]
    missing_df = pd.concat(missing_dfs, axis=1)
    st.dataframe(missing_df)

def show():
    st.title("Advanced Data Insights")

    initialize_openai_client()

    if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
        st.warning("Please upload data files first in the 'Upload Data' page.")
        return

    # File selection
    file_names = [file['name'] for file in st.session_state.uploaded_files]
    selected_files = st.multiselect("Select files to analyze:", file_names, default=[file_names[0]])

    if not selected_files:
        st.warning("Please select at least one file to analyze.")
        return

    dfs = []
    for file_name in selected_files:
        file_data = next(file for file in st.session_state.uploaded_files if file['name'] == file_name)
        df = pd.read_csv(BytesIO(file_data['content'])) if file_data['type'] == '.csv' else pd.read_excel(BytesIO(file_data['content']))
        dfs.append(df)

    # Create two columns: main content and chat
    col1, col2 = st.columns([3, 1], gap="small")
    st.markdown("<style> .block-container { max-width: 90%; } </style>", unsafe_allow_html=True)

    with col1:
        # Data Summary for each selected file
        for i, df in enumerate(dfs):
            st.subheader(f"Data Summary: {selected_files[i]}")
            show_data_summary(df)

        # Comparative Insights
        if len(dfs) > 1:
            st.subheader("Comparative Insights")
            show_comparative_insights(dfs, selected_files)

        # Interactive Visualizations
        plot_interactive_viz(dfs, selected_files)

        # AI-Generated Insights
        st.header("AI-Generated Insights")
        if st.button("Generate AI Insights"):
            with st.spinner("Generating insights..."):
                try:
                    insights = generate_ai_insights(dfs, selected_files)
                    st.markdown(insights)
                except Exception as e:
                    st.error(f"An error occurred while generating insights: {str(e)}")

    with col2:
        st.header("Chat with Your Data")
        st.write("Ask questions about your data and get AI-powered insights!")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Ask about your data"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = generate_ai_response(dfs, prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred while generating the response: {str(e)}")

if __name__ == "__main__":
    show()