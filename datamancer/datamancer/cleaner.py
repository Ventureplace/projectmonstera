# datamancer/cleaner.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold

def smart_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning operations on the input dataframe.
    
    :param df: Input dataframe
    :return: Cleaned dataframe
    """
    # Make a copy of the dataframe to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with median
    numeric_imputer = SimpleImputer(strategy='median')
    cleaned_df[numeric_columns] = numeric_imputer.fit_transform(cleaned_df[numeric_columns])
    
    # Impute categorical columns with mode
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    cleaned_df[categorical_columns] = categorical_imputer.fit_transform(cleaned_df[categorical_columns])
    
    # Handle outliers using IQR method
    for column in numeric_columns:
        Q1 = cleaned_df[column].quantile(0.25)
        Q3 = cleaned_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df[column] = cleaned_df[column].clip(lower_bound, upper_bound)
    
    return cleaned_df

def smart_clean_extended(df: pd.DataFrame, variance_threshold: float = 0.1, skew_threshold: float = 0.5) -> pd.DataFrame:
    """
    Perform extended cleaning operations on the input dataframe.
    
    :param df: Input dataframe
    :param variance_threshold: Threshold for removing low variance features
    :param skew_threshold: Threshold for handling skewed features
    :return: Cleaned dataframe with advanced preprocessing
    """
    # Start with basic cleaning
    cleaned_df = smart_clean(df)
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=variance_threshold)
    cleaned_df = pd.DataFrame(selector.fit_transform(cleaned_df), 
                              columns=cleaned_df.columns[selector.get_support()],
                              index=cleaned_df.index)
    
    # Handle skewed features
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if abs(cleaned_df[column].skew()) > skew_threshold:
            cleaned_df[column] = np.log1p(cleaned_df[column] - cleaned_df[column].min() + 1)
    
    # Scale numeric features
    scaler = RobustScaler()  # RobustScaler is less sensitive to outliers
    cleaned_df[numeric_columns] = scaler.fit_transform(cleaned_df[numeric_columns])
    
    # Encode categorical variables
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
    cleaned_df = pd.get_dummies(cleaned_df, columns=categorical_columns, drop_first=True)
    
    return cleaned_df

def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Remove features with low variance.
    
    :param df: Input dataframe
    :param threshold: Variance threshold
    :return: Dataframe with low variance features removed
    """
    selector = VarianceThreshold(threshold=threshold)
    return pd.DataFrame(selector.fit_transform(df), 
                        columns=df.columns[selector.get_support()],
                        index=df.index)

def handle_skewed_features(df: pd.DataFrame, skew_threshold: float = 0.5) -> pd.DataFrame:
    """
    Apply log transformation to highly skewed numerical features.
    
    :param df: Input dataframe
    :param skew_threshold: Skewness threshold
    :return: Dataframe with skewed features transformed
    """
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    skewed_features = df[numeric_features].apply(lambda x: x.skew()).abs() > skew_threshold
    df[numeric_features.loc[skewed_features]] = np.log1p(df[numeric_features.loc[skewed_features]])
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing special characters and spaces.
    
    :param df: Input dataframe
    :return: Dataframe with cleaned column names
    """
    df.columns = df.columns.str.replace('[^\w\s]', '', regex=True).str.replace('\s+', '_', regex=True).str.lower()
    return df