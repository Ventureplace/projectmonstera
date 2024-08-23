# type_infer.py

from enum import Enum

class dtype(Enum):
    integer = 'integer'
    float = 'float'
    datetime = 'datetime'
    date = 'date'
    categorical = 'categorical'
    binary = 'binary'
    num_array = 'num_array'
    cat_array = 'cat_array'
    tags = 'tags'
    short_text = 'short_text'
    rich_text = 'rich_text'
    quantity = 'quantity'
    num_tsarray = 'num_tsarray'
    cat_tsarray = 'cat_tsarray'
    invalid = 'invalid'

def infer_types(df, config=None):
    dtypes = {}
    for column in df.columns:
        if df[column].dtype == 'int64':
            dtypes[column] = dtype.integer
        elif df[column].dtype == 'float64':
            dtypes[column] = dtype.float
        elif df[column].dtype == 'object':
            if df[column].nunique() / len(df[column]) < 0.5:  # arbitrary threshold
                dtypes[column] = dtype.categorical
            else:
                dtypes[column] = dtype.short_text
        elif df[column].dtype == 'bool':
            dtypes[column] = dtype.binary
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            dtypes[column] = dtype.datetime
        else:
            dtypes[column] = dtype.invalid
    
    return dtypes

class TypeInformation:
    def __init__(self, dtypes, identifiers=None):
        self.dtypes = dtypes
        self.identifiers = identifiers or {}