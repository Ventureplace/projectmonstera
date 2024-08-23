# datamancer/splitter.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from typing import Tuple, List, Union
from type_infer import infer_types, TypeInformation

def simple_split(df: pd.DataFrame, 
                 target_column: str, 
                 test_size: float = 0.2, 
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a simple random split of the data.

    :param df: Input dataframe
    :param target_column: Name of the target column
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random state for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def stratified_split(df: pd.DataFrame, 
                     target_column: str, 
                     test_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified split of the data, maintaining the percentage of samples for each class.

    :param df: Input dataframe
    :param target_column: Name of the target column
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random state for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def time_series_split(df: pd.DataFrame, 
                      target_column: str, 
                      date_column: str,
                      test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a time series split of the data.

    :param df: Input dataframe
    :param target_column: Name of the target column
    :param date_column: Name of the date column
    :param test_size: Proportion of the dataset to include in the test split
    :return: X_train, X_test, y_train, y_test
    """
    df = df.sort_values(by=date_column)
    split_index = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    
    X_train = train.drop(columns=[target_column, date_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column, date_column])
    y_test = test[target_column]
    
    return X_train, X_test, y_train, y_test

def group_split(df: pd.DataFrame, 
                target_column: str, 
                group_column: str,
                test_size: float = 0.2, 
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a group-based split of the data, ensuring that all data for a group is either in train or test.

    :param df: Input dataframe
    :param target_column: Name of the target column
    :param group_column: Name of the column containing group information
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random state for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    groups = df[group_column].unique()
    np.random.seed(random_state)
    test_groups = np.random.choice(groups, size=int(len(groups) * test_size), replace=False)
    
    test = df[df[group_column].isin(test_groups)]
    train = df[~df[group_column].isin(test_groups)]
    
    X_train = train.drop(columns=[target_column, group_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column, group_column])
    y_test = test[target_column]
    
    return X_train, X_test, y_train, y_test

def k_fold_split(df: pd.DataFrame, 
                 target_column: str, 
                 n_splits: int = 5, 
                 shuffle: bool = True, 
                 random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Perform k-fold cross-validation split of the data.

    :param df: Input dataframe
    :param target_column: Name of the target column
    :param n_splits: Number of splits (folds)
    :param shuffle: Whether to shuffle the data before splitting
    :param random_state: Random state for reproducibility
    :return: List of (X_train, X_test, y_train, y_test) for each fold
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    splits = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((X_train, X_test, y_train, y_test))
    
    return splits

def time_series_cv_split(df: pd.DataFrame, 
                         target_column: str, 
                         date_column: str,
                         n_splits: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Perform time series cross-validation split of the data.

    :param df: Input dataframe
    :param target_column: Name of the target column
    :param date_column: Name of the date column
    :param n_splits: Number of splits
    :return: List of (X_train, X_test, y_train, y_test) for each split
    """
    df = df.sort_values(by=date_column)
    X = df.drop(columns=[target_column, date_column])
    y = df[target_column]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    splits = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((X_train, X_test, y_train, y_test))
    
    return splits

def smart_split(df: pd.DataFrame, 
                target_column: str, 
                split_type: str = 'simple',
                test_size: float = 0.2,
                date_column: str = None,
                group_column: str = None,
                n_splits: int = 5,
                random_state: int = 42) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], 
                                                 List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]]:
    """
    Perform a smart split of the data based on the specified split type.

    :param df: Input dataframe
    :param target_column: Name of the target column
    :param split_type: Type of split to perform ('simple', 'stratified', 'time_series', 'group', 'k_fold', 'time_series_cv')
    :param test_size: Proportion of the dataset to include in the test split (for non-CV splits)
    :param date_column: Name of the date column (for time series splits)
    :param group_column: Name of the column containing group information (for group split)
    :param n_splits: Number of splits for k-fold and time series cross-validation
    :param random_state: Random state for reproducibility
    :return: Split data according to the specified split type
    """
    if split_type == 'simple':
        return simple_split(df, target_column, test_size, random_state)
    elif split_type == 'stratified':
        return stratified_split(df, target_column, test_size, random_state)
    elif split_type == 'time_series':
        if date_column is None:
            raise ValueError("date_column must be specified for time series split")
        return time_series_split(df, target_column, date_column, test_size)
    elif split_type == 'group':
        if group_column is None:
            raise ValueError("group_column must be specified for group split")
        return group_split(df, target_column, group_column, test_size, random_state)
    elif split_type == 'k_fold':
        return k_fold_split(df, target_column, n_splits, True, random_state)
    elif split_type == 'time_series_cv':
        if date_column is None:
            raise ValueError("date_column must be specified for time series cross-validation split")
        return time_series_cv_split(df, target_column, date_column, n_splits)
    else:
        raise ValueError("Invalid split_type. Choose from 'simple', 'stratified', 'time_series', 'group', 'k_fold', 'time_series_cv'")