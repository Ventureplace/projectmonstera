# dashboard/app.py
import streamlit as st
from pages import data_upload, data_cleaning, feature_engineering, data_insights, model_preparation, dashboard_summary
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Datamancer", layout="wide")

# Custom CSS to make it look more like Salesforce CRM and fix the width issue
st.markdown("""
<style>
.main .block-container {
    max-width: 100%;
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 1rem;
}
.uploadedFile {
    display: none;
}
.stExpander {
    background-color: #f0f2f6;
    border-radius: 0.5rem;
    margin-top: 1rem;
}
.stExpander > div:first-child {
    border-radius: 0.5rem 0.5rem 0 0;
}
.stExpander > div:last-child {
    border-radius: 0 0 0.5rem 0.5rem;
}
.stButton > button {
    border: none;
    background: none;
    color: #4CAF50;  /* You can change this color */
    font-weight: bold;
    padding: 0;
}
.stButton > button:hover {
    color: #45a049;  /* Darker shade for hover effect */
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Datamancer")
st.sidebar.image("dashboard/static/logo.png", width=200)
page = st.sidebar.radio("Navigate", ["Upload Data", "Clean Data", "Engineer Features", "Data Insights", "Prepare Model", "Dashboard Summary"])

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
elif page == "Dashboard Summary":
    dashboard_summary.show()