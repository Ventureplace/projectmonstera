import streamlit as st
import pandas as pd
import sys
import os
import io
import sqlite3
import json
import xml.etree.ElementTree as ET
from openpyxl import load_workbook
from docx import Document
import PyPDF2
import configparser
import yaml
import csv

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from datamancer.validator import validate_data, DataSchema
    from datamancer.type_infer import infer_types, TypeInformation
    from datamancer.validator import validate_data_robust
    VALIDATOR_AVAILABLE = True
except ImportError:
    print("Warning: DataSchema and validate_data are not available. Data validation will be skipped.")
    VALIDATOR_AVAILABLE = False

def read_file(file, file_type):
    if file_type in ['.csv', '.txt']:
        return pd.read_csv(io.StringIO(file.getvalue().decode('utf-8')))
    elif file_type in ['.xls', '.xlsx']:
        return pd.read_excel(io.BytesIO(file.getvalue()))
    elif file_type == '.json':
        return pd.read_json(io.StringIO(file.getvalue().decode('utf-8')))
    elif file_type == '.xml':
        tree = ET.parse(io.BytesIO(file.getvalue()))
        root = tree.getroot()
        data = []
        for child in root:
            data.append({subchild.tag: subchild.text for subchild in child})
        return pd.DataFrame(data)
    elif file_type in ['.sql', '.db', '.sqlite']:
        conn = sqlite3.connect(':memory:')
        conn.cursor().executescript(file.getvalue().decode('utf-8'))
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if not tables.empty:
            table_name = tables.iloc[0]['name']
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    elif file_type == '.docx':
        doc = Document(io.BytesIO(file.getvalue()))
        content = [para.text for para in doc.paragraphs]
        return pd.DataFrame({'Content': content})
    elif file_type == '.pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        content = []
        for page in pdf_reader.pages:
            content.append(page.extract_text())
        return pd.DataFrame({'Content': content})
    elif file_type in ['.ini', '.cfg']:
        config = configparser.ConfigParser()
        config.read_string(file.getvalue().decode('utf-8'))
        data = {section: dict(config[section]) for section in config.sections()}
        return pd.DataFrame.from_dict(data, orient='index')
    elif file_type == '.log':
        return pd.read_csv(io.StringIO(file.getvalue().decode('utf-8')), sep='\s+', engine='python')
    elif file_type in ['.yaml', '.yml']:
        return pd.DataFrame(yaml.safe_load(file.getvalue().decode('utf-8')))
    elif file_type == '.tsv':
        return pd.read_csv(io.StringIO(file.getvalue().decode('utf-8')), sep='\t')
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None

def validate_data_robust(data):
    validation_results = {}
    for column in data.columns:
        column_type = data[column].dtype
        non_null_count = data[column].count()
        unique_count = data[column].nunique()
        
        validation_results[column] = {
            "type": str(column_type),
            "non_null_percentage": (non_null_count / len(data)) * 100,
            "unique_percentage": (unique_count / len(data)) * 100,
            "sample_values": data[column].sample(min(5, len(data))).tolist()
        }
    
    return validation_results

def show():
    st.title("Upload and Manage Your Data")

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded_file = st.file_uploader("", type=None, accept_multiple_files=False, help="Limit 200MB per file")

        with st.expander("See supported file types"):
            st.write("""
            - CSV (.csv)
            - Excel (.xls, .xlsx)
            - Word (.doc, .docx)
            - PDF (.pdf)
            - SQL (.sql)
            - SQLite (.db, .sqlite)
            - XML (.xml)
            - JSON (.json)
            - Text (.txt)
            - Config files (.ini, .cfg)
            - Log files (.log)
            - YAML (.yaml, .yml)
            - TSV (.tsv)
            """)

        if uploaded_file is not None:
            file_type = os.path.splitext(uploaded_file.name)[1].lower()
            
            try:
                data = read_file(uploaded_file, file_type)
                if data is not None:
                    st.write(data.head())
                    
                    if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                        st.session_state.uploaded_files.append({
                            'name': uploaded_file.name,
                            'type': file_type,
                            'size': uploaded_file.size,
                            'content': uploaded_file.getvalue()
                        })
                    
                    # Save the data for use in other pages
                    st.session_state.data = data
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

        # Display list of uploaded files
        if st.session_state.uploaded_files:
            st.markdown("<h4 style='font-size: 18px;'>Uploaded Files</h4>", unsafe_allow_html=True)
            for idx, file in enumerate(st.session_state.uploaded_files):
                if st.button(f"View {file['name']}", key=f"view_{idx}"):
                    st.write(f"Viewing: {file['name']}")
                    try:
                        viewed_data = read_file(io.BytesIO(file['content']), file['type'])
                        st.write(viewed_data.head())
                        st.write(f"Shape: {viewed_data.shape}")
                        st.write(f"Columns: {viewed_data.columns.tolist()}")
                        
                        # Update the current data for validation
                        st.session_state.data = viewed_data
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")

    with col2:
        st.markdown('<div class="validation-column">', unsafe_allow_html=True)
        with st.expander("Data Validation", expanded=True):
            if 'data' in st.session_state:
                validation_results = validate_data_robust(st.session_state.data)
                for column, results in validation_results.items():
                    st.write(f"Column: {column}")
                    st.write(f"Type: {results['type']}")
                    st.write(f"Non-null: {results['non_null_percentage']:.2f}%")
                    st.write(f"Unique: {results['unique_percentage']:.2f}%")
                    st.write(f"Sample values: {results['sample_values']}")
                    st.write("---")
            else:
                st.write("Upload or select a file to see validation results.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show()

st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)