import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from openai import OpenAI
import json
from io import BytesIO, StringIO
import base64
from sqlalchemy import create_engine

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
        st.error("Please enter a valid OpenAI API key to use the AI features.")

def generate_sql_script_ai(df: pd.DataFrame, table_name: str, db_type: str = 'sqlite') -> str:
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

def show_sql_generation(df):
    st.subheader("SQL Script Generation")
    
    table_name = st.text_input("Enter table name:", "my_table")
    db_type = st.selectbox("Select database type:", ["sqlite", "mysql", "postgresql"])
    
    if st.button("Generate SQL Script"):
        with st.spinner("Generating SQL script..."):
            try:
                sql_script = generate_sql_script_ai(df, table_name, db_type)
                
                # Wrap the SQL script display in an expander
                with st.expander("Generated SQL Script", expanded=True):
                    st.code(sql_script, language='sql')
                
                # Option to download the SQL script
                st.download_button(
                    label="Download SQL Script",
                    data=sql_script,
                    file_name=f"{table_name}_script.sql",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"An error occurred while generating the SQL script: {str(e)}")


def show_data_summary(df):
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Rows", df.shape[0])
    with col2:
        st.metric("Number of Columns", df.shape[1])

    with st.expander("Column Types"):
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    with st.expander("First Few Rows"):
        st.dataframe(df.head(), use_container_width=True)

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

def plot_quick_viz(df):
    st.subheader("Quick Visualizations")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        selected_cols = st.multiselect("Select numeric columns for visualization:", numeric_cols, default=[numeric_cols[0]])

        if selected_cols:
            viz_type = st.radio("Select visualization type:", ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap"])

            if viz_type == "Histogram":
                fig = px.histogram(df, x=selected_cols[0], marginal="box")
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Box Plot":
                fig = px.box(df, y=selected_cols)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Scatter Plot" and len(selected_cols) >= 2:
                fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1], color=selected_cols[2] if len(selected_cols) > 2 else None)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Correlation Heatmap":
                corr = df[selected_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

def ai_chat(df):
    st.subheader("AI Chat")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your data:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if client:
                data_sample = df.head(5).to_dict(orient='records')
                column_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
                data_summary = df.describe().to_dict()
                
                prompt_context = f"""
                Given the following dataset:

                Data sample:
                {json.dumps(data_sample, indent=2)}

                Column information:
                {json.dumps(column_info, indent=2)}

                Data summary:
                {json.dumps(data_summary, indent=2)}

                Please answer the following question:
                {prompt}

                Provide a clear and concise answer based on the data provided.
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst assistant."},
                        {"role": "user", "content": prompt_context}
                    ],
                    stream=True
                )

                for chunk in response:
                    full_response += chunk.choices[0].delta.content or ""
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            else:
                message_placeholder.markdown("Please provide a valid OpenAI API key to use the AI chat feature.")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

def generate_report(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.describe().to_excel(writer, sheet_name='Summary Statistics')
        df.head(10).to_excel(writer, sheet_name='Data Preview')
        df.isnull().sum().to_excel(writer, sheet_name='Missing Values')
    
    buffer.seek(0)
    return buffer

def show():
    st.set_page_config(layout="wide", page_title="Data Dashboard Summary")

    initialize_openai_client()

    # Custom CSS for improved aesthetics
    st.markdown("""
    <style>
    .stApp {
        max-width: 20000px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .sql-column {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Your Data")

    initialize_openai_client()

    # Progress bar
    progress = st.progress(0)

    if 'data' not in st.session_state:
        st.warning("No data uploaded yet. Please go to the 'Upload Data' page to upload your data.")
        return

    df = st.session_state.data

    # Upload status
    st.sidebar.success(f"Dataset: {st.session_state.get('uploaded_file_name', 'Unknown')}")
    st.sidebar.info(f"Upload time: {st.session_state.get('upload_time', 'Unknown')}")

    # Theme selector
    theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    progress.progress(25)

    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        show_data_summary(df)

    with col2:
        show_quick_insights(df)

    with col3:
        st.markdown('<div class="sql-column">', unsafe_allow_html=True)
        show_sql_generation(df)
        st.markdown('</div>', unsafe_allow_html=True)

    progress.progress(50)

    plot_quick_viz(df)

    progress.progress(75)

    ai_chat(df)

    progress.progress(100)

    st.subheader("Navigate to Other Pages")
    st.info("For more detailed analysis and operations, visit the following pages:")
    st.markdown("- [Upload Data](/Upload_Data) to upload or update your dataset")
    st.markdown("- [Clean Data](/Clean_Data) to perform data cleaning operations")
    st.markdown("- [Data Insights](/Data_Insights) for in-depth data analysis and visualizations")

    # Download report
    report = generate_report(df)
    st.download_button(
        label="Download Data Report",
        data=report,
        file_name="data_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Feedback form
    st.sidebar.subheader("Feedback")
    feedback = st.sidebar.text_area("Share your thoughts or report issues:")
    if st.sidebar.button("Submit Feedback"):
        # Here you would typically send this feedback to a database or email
        st.sidebar.success("Thank you for your feedback!")

if __name__ == "__main__":
    show()
