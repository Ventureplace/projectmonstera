# dashboard/pages/feature_engineering.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from datamancer.feature_engineer import auto_features, create_interaction_features, create_polynomial_features, create_date_features
from type_infer import infer_types, TypeInformation

def show():
    st.title("Feature Engineering")

    if 'data' not in st.session_state:
        st.warning("Please upload data first in the 'Upload Data' page.")
        return

    df = st.session_state.get('cleaned_data', st.session_state.data)
    
    st.subheader("Original Data Preview")
    st.write(df.head())

    st.subheader("Feature Engineering Options")

    # Automated Feature Engineering
    st.write("Automated Feature Engineering")
    if st.checkbox("Use automated feature engineering"):
        target_column = st.selectbox("Select target column:", df.columns)
        max_depth = st.slider("Maximum depth for feature generation", 1, 5, 2)
        feature_types = st.multiselect("Select feature types to generate:", ["numeric", "categorical", "all"], default="all")
        
        if st.button("Generate Features"):
            with st.spinner("Generating features..."):
                new_features = auto_features(df, target_column, max_depth, feature_types[0] if feature_types else "all")
                st.session_state.engineered_data = new_features
                st.success(f"Generated {len(new_features.columns) - len(df.columns)} new features!")
                st.write(new_features.head())

    # Interaction Features
    st.write("Interaction Features")
    if st.checkbox("Create interaction features"):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        interaction_columns = st.multiselect("Select columns for interaction:", numeric_columns)
        if st.button("Create Interactions"):
            with st.spinner("Creating interaction features..."):
                df_interactions = create_interaction_features(df, interaction_columns)
                st.session_state.engineered_data = df_interactions
                st.success(f"Created {len(df_interactions.columns) - len(df.columns)} interaction features!")
                st.write(df_interactions.head())

    # Polynomial Features
    st.write("Polynomial Features")
    if st.checkbox("Create polynomial features"):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        poly_columns = st.multiselect("Select columns for polynomial features:", numeric_columns)
        poly_degree = st.slider("Polynomial degree", 2, 5, 2)
        if st.button("Create Polynomial Features"):
            with st.spinner("Creating polynomial features..."):
                df_poly = create_polynomial_features(df, poly_columns, poly_degree)
                st.session_state.engineered_data = df_poly
                st.success(f"Created {len(df_poly.columns) - len(df.columns)} polynomial features!")
                st.write(df_poly.head())

    # Date Features
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0:
        st.write("Date Features")
        if st.checkbox("Create date features"):
            date_column = st.selectbox("Select date column:", date_columns)
            if st.button("Create Date Features"):
                with st.spinner("Creating date features..."):
                    df_date = create_date_features(df, date_column)
                    st.session_state.engineered_data = df_date
                    st.success(f"Created {len(df_date.columns) - len(df.columns)} date features!")
                    st.write(df_date.head())

    # Feature Scaling
    st.write("Feature Scaling")
    if st.checkbox("Scale features"):
        scaler_type = st.selectbox("Select scaling method:", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        columns_to_scale = st.multiselect("Select columns to scale:", numeric_columns)
        
        if st.button("Scale Features"):
            with st.spinner("Scaling features..."):
                if scaler_type == "StandardScaler":
                    scaler = StandardScaler()
                elif scaler_type == "MinMaxScaler":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()
                
                df_scaled = df.copy()
                df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                st.session_state.engineered_data = df_scaled
                st.success("Features scaled successfully!")
                st.write(df_scaled.head())

    # Feature Selection
    st.write("Feature Selection")
    if st.checkbox("Perform feature selection"):
        method = st.selectbox("Select feature selection method:", ["PCA", "Variance Threshold"])
        
        if method == "PCA":
            n_components = st.slider("Number of components", 1, len(df.columns), min(5, len(df.columns)))
            if st.button("Apply PCA"):
                with st.spinner("Applying PCA..."):
                    numeric_data = df.select_dtypes(include=[np.number])
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(numeric_data)
                    df_pca = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
                    st.session_state.engineered_data = df_pca
                    st.success(f"Reduced features to {n_components} principal components!")
                    st.write(df_pca.head())
                    
                    # Plot explained variance ratio
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
                    ax.set_xlabel('Number of Components')
                    ax.set_ylabel('Cumulative Explained Variance')
                    st.pyplot(fig)
        
        elif method == "Variance Threshold":
            threshold = st.slider("Variance threshold", 0.0, 1.0, 0.1, 0.05)
            if st.button("Apply Variance Threshold"):
                with st.spinner("Applying variance threshold..."):
                    from sklearn.feature_selection import VarianceThreshold
                    selector = VarianceThreshold(threshold=threshold)
                    numeric_data = df.select_dtypes(include=[np.number])
                    selected_features = selector.fit_transform(numeric_data)
                    selected_columns = numeric_data.columns[selector.get_support()]
                    df_selected = pd.DataFrame(data=selected_features, columns=selected_columns)
                    st.session_state.engineered_data = df_selected
                    st.success(f"Selected {len(selected_columns)} features based on variance threshold!")
                    st.write(df_selected.head())

    # Download engineered data
    if 'engineered_data' in st.session_state:
        st.subheader("Download Engineered Data")
        csv = st.session_state.engineered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="engineered_data.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    show()