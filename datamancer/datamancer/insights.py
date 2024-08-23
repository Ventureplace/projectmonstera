# datamancer/insights.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

def generate_data_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive data report."""
    report = {}
    report['num_rows'], report['num_columns'] = df.shape
    report['column_types'] = df.dtypes.to_dict()
    report['missing_values'] = df.isnull().sum().to_dict()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    report['categorical_stats'] = {col: df[col].value_counts().to_dict() for col in categorical_cols}
    
    return report

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix for numerical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr()

def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Plot a heatmap of the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    return plt.gcf()

def get_top_correlated_features(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
    """Get pairs of features with correlation above a specified threshold."""
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    return sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)

def plot_distribution(df: pd.DataFrame, column: str):
    """Plot the distribution of a specified column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    return plt.gcf()

def plot_boxplot(df: pd.DataFrame, column: str):
    """Plot a boxplot for a specified column."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    return plt.gcf()

def feature_importance(df: pd.DataFrame, target_column: str, top_n: int = 10):
    """Calculate and plot feature importances using Random Forest."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if df[target_column].dtype == 'object':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    importances = model.feature_importances_
    
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp)
    plt.title(f'Top {top_n} Feature Importances')
    return plt.gcf()