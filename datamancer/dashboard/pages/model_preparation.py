# dashboard/pages/model_preparation.py

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from datamancer.splitters import smart_split
    from datamancer.type_infer import infer_types, TypeInformation
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    DATAMANCER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing required modules. {str(e)}")
    DATAMANCER_AVAILABLE = False

def show():
    st.title("Model Preparation")

    if 'data' not in st.session_state:
        st.warning("Please upload data first in the 'Upload Data' page.")
        return

    df = st.session_state.get('engineered_data', st.session_state.get('cleaned_data', st.session_state.data))
    
    st.subheader("Data Preview")
    st.write(df.head())

    # Target Selection
    st.subheader("Select Target Variable")
    target_column = st.selectbox("Choose the target variable:", df.columns)
    st.session_state.target = target_column

    # Feature Selection
    st.subheader("Feature Selection")
    feature_selector = st.selectbox("Choose feature selection method:", 
                                    ["All Features", "Select K Best", "Manual Selection"])
    
    if feature_selector == "Select K Best":
        k = st.slider("Select number of features to keep:", 1, len(df.columns)-1, 5)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if df[target_column].dtype == 'object':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(f_regression, k=k)
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
    elif feature_selector == "Manual Selection":
        selected_features = st.multiselect("Select features:", [col for col in df.columns if col != target_column])
    else:
        selected_features = [col for col in df.columns if col != target_column]

    st.write("Selected features:", selected_features)

    # Data Splitting
    st.subheader("Data Splitting")
    split_method = st.selectbox("Choose splitting method:", 
                                ["Simple", "Stratified", "Time Series", "Group"])
    test_size = st.slider("Test set size:", 0.1, 0.5, 0.2)
    
    if split_method == "Time Series":
        date_column = st.selectbox("Select date column for time series split:", 
                                   df.select_dtypes(include=['datetime64']).columns)
    elif split_method == "Group":
        group_column = st.selectbox("Select group column for group split:", df.columns)
    else:
        date_column = None
        group_column = None

    if st.button("Split Data"):
        X = df[selected_features]
        y = df[target_column]
        
        if DATAMANCER_AVAILABLE:
            split_params = {
                "df": df,
                "target_column": target_column,
                "split_type": split_method.lower(),
                "test_size": test_size,
            }
            
            if date_column:
                split_params["date_column"] = date_column
            if group_column:
                split_params["group_column"] = group_column

            try:
                split_result = smart_split(**split_params)
                if isinstance(split_result, tuple):
                    X_train, X_test, y_train, y_test = split_result
                else:  # It's a list of splits for cross-validation
                    X_train, X_test, y_train, y_test = split_result[0]  # Use the first split
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success("Data split successfully!")
                st.write(f"Training set shape: {X_train.shape}")
                st.write(f"Test set shape: {X_test.shape}")
            except Exception as e:
                st.error(f"Error in splitting data: {str(e)}")
        else:
            st.warning("Advanced splitting methods are not available. Using simple train-test split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("Data split successfully!")
            st.write(f"Training set shape: {X_train.shape}")
            st.write(f"Test set shape: {X_test.shape}")

    # Model Selection
    st.subheader("Model Selection")
    problem_type = "Classification" if df[target_column].dtype == 'object' else "Regression"
    st.write(f"Problem type: {problem_type}")
    
    if problem_type == "Classification":
        model_options = ["Random Forest", "Logistic Regression", "SVM"]
    else:
        model_options = ["Random Forest", "Linear Regression", "SVR"]
    
    selected_model = st.selectbox("Choose a model:", model_options)

    # Model Training
    if st.button("Train Model"):
        if 'X_train' not in st.session_state:
            st.error("Please split the data first.")
            return

        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        if problem_type == "Classification":
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        if selected_model == "Random Forest":
            if problem_type == "Classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif selected_model in ["Logistic Regression", "Linear Regression"]:
            if problem_type == "Classification":
                model = LogisticRegression(random_state=42)
            else:
                model = LinearRegression()
        else:  # SVM
            if problem_type == "Classification":
                model = SVC(random_state=42)
            else:
                model = SVR()

        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
            st.session_state.model = model

        # Model Evaluation
        y_pred = model.predict(X_test)
        if problem_type == "Classification":
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R-squared Score: {r2:.4f}")

        # Feature Importance (for Random Forest)
        if selected_model == "Random Forest":
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
            plt.title("Top 10 Feature Importances")
            st.pyplot(fig)

        # Predictions vs Actual plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        plt.title("Predictions vs Actual")
        st.pyplot(fig)

if __name__ == "__main__":
    show()