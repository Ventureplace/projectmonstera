# dashboard/pages/data_upload.py
import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from datamancer.validator import validate_data, DataSchema
    from datamancer.type_infer import infer_types, TypeInformation
    VALIDATOR_AVAILABLE = True
except ImportError:
    print("Warning: DataSchema and validate_data are not available. Data validation will be skipped.")
    VALIDATOR_AVAILABLE = False

def show():
    st.title("Upload Your Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        
        if VALIDATOR_AVAILABLE:
            st.subheader("Data Schema")
            schemas = []
            for column in data.columns:
                col_type = str(data[column].dtype)
                schema = DataSchema(
                    column_name=column,
                    data_type=col_type,
                    allow_null=st.checkbox(f"Allow null values in {column}"),
                    unique=st.checkbox(f"Unique values in {column}")
                )
                schemas.append(schema)
            
            if st.button("Validate Data"):
                try:
                    validate_data(data, schemas)
                    st.success("Data validation successful!")
                except AssertionError as e:
                    st.error(f"Data validation failed: {str(e)}")
        else:
            st.warning("Data validation is not available due to missing dependencies.")
        
        # Save the data for use in other pages
        st.session_state.data = data

if __name__ == "__main__":
    show()