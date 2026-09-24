"""Data loader module for Automated Anomaly Explanation with Counterfactuals.

Responsibilities:
- Load the raw industrial predictive-maintenance dataset safely without modification.
- Validate dataset schema, types, integrity, and data quality.
- Provide a clean data-quality summary.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

# Default dataset path relative to repository root
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ai4i2020.csv"

# Exact expected schema from docs/DATA_DICTIONARY.md and config/features.yaml
EXPECTED_COLUMNS: List[str] = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# Physical operational measurements used by the detector
NUMERICAL_OPERATIONAL_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Evaluation labels (ground truth failure flags)
LABEL_COLUMNS: List[str] = [
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# String-like categorical and identifier columns
STRING_LIKE_COLUMNS: List[str] = [
    "Product ID",
    "Type",
]


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate dataset structure, expected schema, column types, and data quality.

    Raises clear, actionable exceptions if validation rules are violated.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataframe to validate.

    Returns
    -------
    bool
        True if all validations pass.

    Raises
    ------
    ValueError
        If the dataframe is empty, has missing/unexpected/duplicate columns,
        contains missing values, or contains duplicate rows.
    TypeError
        If numerical columns are not numeric or string-like columns are not string/object.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, received: {type(df).__name__}")

    # Check for empty dataframe
    if df.empty:
        raise ValueError("Dataset validation failed: The dataframe is empty. Expected at least one row.")

    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        seen = set()
        duplicates = set()
        for col in df.columns:
            if col in seen:
                duplicates.add(col)
            seen.add(col)
        raise ValueError(
            f"Dataset validation failed: Duplicate column names detected: {sorted(list(duplicates))}"
        )

    # Check for missing required columns
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset validation failed: Missing expected columns: {missing_columns}"
        )

    # Check for unexpected extra columns
    unexpected_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if unexpected_columns:
        raise ValueError(
            f"Dataset validation failed: Unexpected columns present: {unexpected_columns}"
        )

    # Check for missing values
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        missing_by_col = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        raise ValueError(
            f"Dataset validation failed: Found {total_missing} missing value(s) in columns: {missing_by_col}"
        )

    # Check for duplicate rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(
            f"Dataset validation failed: Found {duplicate_count} duplicate row(s)."
        )

    # Check that operational numerical fields are numeric
    for col in NUMERICAL_OPERATIONAL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Dataset validation failed: Operational feature '{col}' must be numeric, but has dtype '{df[col].dtype}'."
            )

    # Check that evaluation labels are numeric
    for col in LABEL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Dataset validation failed: Evaluation label '{col}' must be numeric, but has dtype '{df[col].dtype}'."
            )

    # Check that UDI identifier is numeric
    if not pd.api.types.is_numeric_dtype(df["UDI"]):
        raise TypeError(
            f"Dataset validation failed: Identifier 'UDI' must be numeric, but has dtype '{df['UDI'].dtype}'."
        )

    # Check that identifier/categorical fields are string-like
    for col in STRING_LIKE_COLUMNS:
        if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
            raise TypeError(
                f"Dataset validation failed: Column '{col}' must be string-like, but has dtype '{df[col].dtype}'."
            )

    return True


def get_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a clean data quality summary dictionary for the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to inspect.

    Returns
    -------
    dict
        Summary containing row_count, column_count, missing_value_count,
        duplicate_row_count, and detected_column_names.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, received: {type(df).__name__}")

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_value_count": int(df.isnull().sum().sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
        "detected_column_names": list(df.columns),
    }


def load_raw_dataset(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and validate the raw AI4I dataset from disk without altering it.

    Parameters
    ----------
    csv_path : str, Path, or None, optional
        Path to the raw CSV file. If None, uses DEFAULT_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Validated pandas DataFrame containing the raw data.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path.
    ValueError
        If the path is not a file or if dataset validation fails.
    TypeError
        If dataset column types fail validation.
    """
    resolved_path = DEFAULT_DATA_PATH if csv_path is None else Path(csv_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at: '{resolved_path}'. "
            "Please ensure the dataset exists in data/raw/ai4i2020.csv."
        )

    if not resolved_path.is_file():
        raise ValueError(f"Specified path is not a valid file: '{resolved_path}'.")

    # Load dataset safely in read-only manner
    df = pd.read_csv(resolved_path)

    # Validate against expected schema, types, and quality rules
    validate_dataset(df)

    return df
