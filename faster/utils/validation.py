"""Validation utility functions."""

from typing import Any, List, Optional, Union
import logging

import pandas as pd
import numpy as np

from faster.utils.logging import get_logger

logger = get_logger(__name__)


def validate_dataframe(
    data: pd.DataFrame,
    target_column: str,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 10,
    categorical_columns: Optional[List[str]] = None,
) -> None:
    """Validate input DataFrame.

    Args:
        data: Input DataFrame to validate
        target_column: Name of target variable
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        categorical_columns: Optional list of categorical column names

    Raises:
        ValueError: If validation fails
    """
    if data is None:
        raise ValueError("Data cannot be None")

    if not target_column:
        raise ValueError("Target column cannot be empty")

    if data.empty:
        raise ValueError("DataFrame cannot be empty")

    if len(data) < min_rows:
        raise ValueError(f"Input DataFrame must have at least {min_rows} rows, got {len(data)}")

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")

    if data.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names")

    null_cols = data.columns[data.isnull().all()].tolist()
    if null_cols:
        raise ValueError(f"Found columns with all null values: {null_cols}")

    if categorical_columns:
        invalid_cols = [col for col in categorical_columns if col not in data.columns]
        if invalid_cols:
            raise ValueError(f"Categorical columns not found in DataFrame: {invalid_cols}")

    if data[target_column].isnull().any():
        raise ValueError("Target column contains missing values")

    _check_data_quality(data, target_column)


def validate_feature_names(feature_names: List[str]) -> None:
    """Validate feature names.

    Args:
        feature_names: List of feature names to validate

    Raises:
        ValueError: If validation fails
    """
    if not feature_names:
        raise ValueError("Feature names list is empty")

    duplicates = {name for name in feature_names if feature_names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Duplicate feature names found: {duplicates}")

    invalid_chars = set("!@#$%^&*()[]{};:,/<>?\\|`~")
    for name in feature_names:
        if any(char in invalid_chars for char in name):
            raise ValueError(f"Invalid characters in feature name: {name}")


def validate_numeric_value(
    value: Union[int, float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_zero: bool = True,
    allow_negative: bool = False,
    param_name: str = "value",
) -> None:
    """Validate numeric value.

    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_zero: Whether zero is allowed
        allow_negative: Whether negative values are allowed
        param_name: Name of parameter for error messages

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{param_name} must be numeric, got {type(value)}")

    if not allow_negative and value < 0:
        raise ValueError(f"{param_name} cannot be negative")

    if not allow_zero and value == 0:
        raise ValueError(f"{param_name} cannot be zero")

    if min_value is not None and value < min_value:
        raise ValueError(f"{param_name} must be >= {min_value}")

    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be <= {max_value}")


def _check_data_quality(data: pd.DataFrame, target_column: str) -> None:
    """Check for potential data quality issues and log warnings."""

    for col in data.select_dtypes(include=["object"]).columns:
        unique_ratio = data[col].nunique() / len(data)
        if unique_ratio > 0.9:
            logger.warning(
                f"Column '{col}' has high cardinality ({data[col].nunique()} unique values)"
            )

    for col in data.select_dtypes(include=np.number).columns:
        if col != target_column:
            skewness = data[col].skew()
            if abs(skewness) > 3:
                logger.warning(f"Column '{col}' is highly skewed (skewness = {skewness:.2f})")

    missing_ratios = data.isnull().mean()
    high_missing = missing_ratios[missing_ratios > 0.2]
    if not high_missing.empty:
        for col, ratio in high_missing.items():
            logger.warning(f"Column '{col}' has {ratio:.1%} missing values")

    for col in data.columns:
        if col != target_column:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.01:
                logger.warning(
                    f"Column '{col}' has low variance ({data[col].nunique()} unique values)"
                )
