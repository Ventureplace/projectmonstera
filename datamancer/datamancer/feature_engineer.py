# datamancer/feature_engineer.py

import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from type_infer import infer_types, TypeInformation

def auto_features(df: pd.DataFrame, target_column: str, max_depth: int = 2, 
                  feature_types: str = "all") -> pd.DataFrame:
    """
    Automatically generate features using featuretools.
    
    :param df: Input dataframe
    :param target_column: Name of the target column
    :param max_depth: Maximum depth for feature generation
    :param feature_types: Types of features to generate ("all", "numeric", or "categorical")
    :return: Dataframe with generated features
    """
    # Create an entity set
    es = ft.EntitySet(id="dataset")
    es = es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="index",
        make_index=True,
        time_index=None
    )
    
    # Define primitive types based on feature_types
    if feature_types == "numeric":
        primitive_types = ["numeric"]
    elif feature_types == "categorical":
        primitive_types = ["categorical"]
    else:
        primitive_types = None
    
    # Generate features
    feature_matrix, feature_defs = ft.dfs(entityset=es, 
                                          target_dataframe_name="data",
                                          trans_primitives=["add_numeric", "multiply_numeric", "divide_numeric"],
                                          agg_primitives=["sum", "std", "max", "min", "mean", "count", "percent_true"],
                                          max_depth=max_depth,
                                          ignore_columns=[target_column],
                                          feature_types=primitive_types)
    
    return feature_matrix

def create_interaction_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Create interaction features between specified columns.
    
    :param df: Input dataframe
    :param columns: List of column names to create interactions for
    :return: Dataframe with interaction features added
    """
    interactions = df[columns].copy()
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col_name = f"{columns[i]}_{columns[j]}_interaction"
            interactions[col_name] = df[columns[i]] * df[columns[j]]
    return pd.concat([df, interactions], axis=1)

def create_polynomial_features(df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    :param df: Input dataframe
    :param columns: List of column names to create polynomial features for
    :param degree: Degree of polynomial features
    :return: Dataframe with polynomial features added
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[columns])
    poly_feature_names = poly.get_feature_names(columns)
    
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    return pd.concat([df, poly_df], axis=1)

def create_pca_features(df: pd.DataFrame, columns: list, n_components: int = 2) -> pd.DataFrame:
    """
    Create PCA features for specified columns.
    
    :param df: Input dataframe
    :param columns: List of column names to create PCA features for
    :param n_components: Number of PCA components to generate
    :return: Dataframe with PCA features added
    """
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(df[columns])
    pca_feature_names = [f"PCA_{i+1}" for i in range(n_components)]
    
    pca_df = pd.DataFrame(pca_features, columns=pca_feature_names, index=df.index)
    return pd.concat([df, pca_df], axis=1)

def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create date-based features from a date column.
    
    :param df: Input dataframe
    :param date_column: Name of the date column
    :return: Dataframe with date features added
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    df[f'{date_column}_is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    return df

def feature_engineer(df: pd.DataFrame, target_column: str, 
                     numeric_columns: list = None, 
                     categorical_columns: list = None,
                     date_columns: list = None,
                     interaction_columns: list = None,
                     polynomial_columns: list = None,
                     pca_columns: list = None,
                     auto_feature_depth: int = 2) -> pd.DataFrame:
    """
    Perform comprehensive feature engineering on the input dataframe.
    
    :param df: Input dataframe
    :param target_column: Name of the target column
    :param numeric_columns: List of numeric column names
    :param categorical_columns: List of categorical column names
    :param date_columns: List of date column names
    :param interaction_columns: List of columns to create interactions for
    :param polynomial_columns: List of columns to create polynomial features for
    :param pca_columns: List of columns to create PCA features for
    :param auto_feature_depth: Maximum depth for automated feature generation
    :return: Dataframe with engineered features
    """
    # Automated feature generation
    df = auto_features(df, target_column, max_depth=auto_feature_depth)
    
    # Create interaction features
    if interaction_columns:
        df = create_interaction_features(df, interaction_columns)
    
    # Create polynomial features
    if polynomial_columns:
        df = create_polynomial_features(df, polynomial_columns)
    
    # Create PCA features
    if pca_columns:
        df = create_pca_features(df, pca_columns)
    
    # Create date features
    if date_columns:
        for date_column in date_columns:
            df = create_date_features(df, date_column)
    
    return df