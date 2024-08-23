# datamancer/helpers.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from type_infer import infer_types, TypeInformation

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> Tuple[pd.Series, pd.Series]:
    """
    Detect outliers in a specified column using either IQR or Z-score method.
    
    :param df: Input dataframe
    :param column: Column to detect outliers in
    :param method: Method to use for outlier detection ('iqr' or 'zscore')
    :return: Tuple of two Series: lower and upper bounds for outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    return lower_bound, upper_bound

def correlation_matrix(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Compute the correlation matrix for specified columns.
    
    :param df: Input dataframe
    :param columns: List of columns to compute correlations for. If None, use all numeric columns.
    :return: Correlation matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    return df[columns].corr()

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)):
    """
    Plot a heatmap of the correlation matrix.
    
    :param corr_matrix: Correlation matrix
    :param figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.show()

def get_top_correlated_features(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Get pairs of features with correlation above a specified threshold.
    
    :param corr_matrix: Correlation matrix
    :param threshold: Correlation threshold
    :return: List of tuples (feature1, feature2, correlation)
    """
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    return sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)

def plot_feature_importances(feature_names: List[str], importances: List[float], top_n: int = 10):
    """
    Plot feature importances.
    
    :param feature_names: List of feature names
    :param importances: List of feature importances
    :param top_n: Number of top features to plot
    """
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp)
    plt.title(f'Top {top_n} Feature Importances')
    plt.show()

def plot_distribution(df: pd.DataFrame, column: str):
    """
    Plot the distribution of a specified column.
    
    :param df: Input dataframe
    :param column: Column to plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_boxplot(df: pd.DataFrame, column: str):
    """
    Plot a boxplot for a specified column.
    
    :param df: Input dataframe
    :param column: Column to plot
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()