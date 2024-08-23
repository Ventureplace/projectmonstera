# datamancer/validator.py

from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Callable
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype
from .type_infer import infer_types, TypeInformation

class ColumnSchema(BaseModel):
    name: str
    dtype: str
    nullable: bool = False
    unique: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Union[str, int, float]]] = None
    regex_pattern: Optional[str] = None
    date_format: Optional[str] = None

    @classmethod
    def check_dtype(cls, v):
        allowed_dtypes = ['int', 'float', 'str', 'bool', 'datetime']
        if v not in allowed_dtypes:
            raise ValueError(f"dtype must be one of {allowed_dtypes}")
        return v

class DataSchema(BaseModel):
    columns: List[ColumnSchema]
    row_count: Optional[int] = None
    custom_validators: Optional[Dict[str, Callable]] = Field(default_factory=dict)

# Rest of the file remains the same