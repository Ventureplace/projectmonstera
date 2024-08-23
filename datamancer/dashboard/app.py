# dashboard/app.py
import streamlit as st
from pages import data_upload, data_cleaning, feature_engineering, data_insights, model_preparation
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Datamancer", layout="wide")

# Custom CSS to make it look more like Salesforce CRM
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f6f9;
    }
    .stButton>button {
        color: #fff;
        background-color: #0070d2;
        border-radius: 4px;
    }
    .stTextInput>div>div>input {
        background-color: #fff;
        color: #16325c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for navigation
st.sidebar.title("Datamancer")
st.sidebar.image("dashboard/static/logo.png", width=200)
page = st.sidebar.radio("Navigate", ["Upload Data", "Clean Data", "Engineer Features", "Data Insights", "Prepare Model"])

# Main content
if page == "Upload Data":
    data_upload.show()
elif page == "Clean Data":
    data_cleaning.show()
elif page == "Engineer Features":
    feature_engineering.show()
elif page == "Data Insights":
    data_insights.show()
elif page == "Prepare Model":
    model_preparation.show()