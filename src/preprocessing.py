"""Data preprocessing module for Automated Anomaly Explanation with Counterfactuals.

Responsibilities:
- cleaning
- encoding/scaling
- preprocessing pipeline persistence
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import yaml
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import the data loader validation function
from src.data_loader import validate_dataset

# Resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES_CONFIG = PROJECT_ROOT / "config" / "features.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "config" / "model_config.yaml"


class DataPreprocessor:
    """Handles dataset splitting, scaling, and feature selection for the anomaly detector.

    Reads configuration from YAML files to ensure consistency with approved project policies.
    Maintains a fitted StandardScaler for the approved detector input features.
    """

    def __init__(
        self,
        features_config_path: Optional[Path] = None,
        model_config_path: Optional[Path] = None,
    ):
        """Initialize the preprocessor using the provided or default configuration paths.

        Parameters
        ----------
        features_config_path : Optional[Path]
            Path to features.yaml configuration file.
        model_config_path : Optional[Path]
            Path to model_config.yaml configuration file.
        """
        self.features_config_path = features_config_path or DEFAULT_FEATURES_CONFIG
        self.model_config_path = model_config_path or DEFAULT_MODEL_CONFIG

        # Load configurations
        with open(self.features_config_path, "r") as f:
            self.features_config = yaml.safe_load(f)

        with open(self.model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)

        # Extract required config values
        self.feature_names = self.features_config["feature_sets"]["detector_inputs"]
        
        self.label_col = self.model_config["data_split"]["known_normal_filter"]["column"]
        self.normal_val = self.model_config["data_split"]["known_normal_filter"]["value"]
        self.failure_val = self.model_config["data_split"]["known_failure_filter"]["value"]

        self.train_fraction = self.model_config["data_split"]["normal_training_fraction"]
        self.random_state = self.model_config["data_split"]["random_state"]
        self.shuffle = self.model_config["data_split"]["shuffle"]

        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_datasets(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the validated raw dataset into training and evaluation partitions.

        Partitions:
        - Normal training records
        - Held-out normal evaluation records
        - Known machine-failure evaluation records

        Parameters
        ----------
        df : pd.DataFrame
            The raw pandas DataFrame loaded by data_loader.py.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (normal_train, normal_eval, failure_eval) DataFrames containing all original columns.
        """
        # Validate the incoming dataframe
        validate_dataset(df)

        # Filter known normal and failure records
        normal_df = df[df[self.label_col] == self.normal_val]
        failure_eval = df[df[self.label_col] == self.failure_val]

        # Split normal records into training and evaluation sets
        normal_train, normal_eval = train_test_split(
            normal_df,
            train_size=self.train_fraction,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        return normal_train, normal_eval, failure_eval

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit the StandardScaler on the detector input features of the provided DataFrame.
        
        This method accepts only the normal-training partition created by
        prepare_datasets(). It enforces the approved training policy.

        Parameters
        ----------
        df : pd.DataFrame
            The normal training DataFrame containing the unscaled original data.

        Returns
        -------
        DataPreprocessor
            The fitted preprocessor instance.
        
        Raises
        ------
        ValueError
            If the input contains any known machine-failure records.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
            
        validate_dataset(df)
        
        if not (df[self.label_col] == self.normal_val).all():
            raise ValueError("Training data contains known machine-failure records.")

        # Select only the approved features
        features = df[self.feature_names]
        self.scaler.fit(features)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale the detector input features using the fitted StandardScaler.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the original unscaled features.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the scaled detector input features,
            preserving the exact approved feature order and original index.

        Raises
        ------
        RuntimeError
            If called before the preprocessor has been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        features = df[self.feature_names]
        scaled_array = self.scaler.transform(features)

        # Return as DataFrame with the original index and feature names
        return pd.DataFrame(
            scaled_array, index=features.index, columns=self.feature_names
        )

    def inverse_transform(self, scaled_df: pd.DataFrame) -> pd.DataFrame:
        """Convert scaled detector input features back into original units.

        Parameters
        ----------
        scaled_df : pd.DataFrame
            DataFrame containing scaled feature values.

        Returns
        -------
        pd.DataFrame
            DataFrame containing unscaled feature values in original units.

        Raises
        ------
        RuntimeError
            If called before the preprocessor has been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling inverse_transform.")

        # Ensure column order matches exactly
        scaled_features = scaled_df[self.feature_names]
        unscaled_array = self.scaler.inverse_transform(scaled_features)

        return pd.DataFrame(
            unscaled_array, index=scaled_features.index, columns=self.feature_names
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Save the fitted preprocessor object to disk using joblib.

        Parameters
        ----------
        path : Optional[Path]
            Path where the preprocessor should be saved. If None, uses the
            artifact path defined in model_config.yaml.

        Raises
        ------
        RuntimeError
            If called before the preprocessor has been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")

        if path is None:
            # Resolve default path from config
            relative_path = self.model_config["artifacts"]["scaler_path"]
            path = PROJECT_ROOT / relative_path
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        
        # We save a dictionary containing both the scaler and feature names
        # to ensure consistency upon loading
        artifact = {
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        joblib.dump(artifact, path)

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        features_config_path: Optional[Path] = None,
        model_config_path: Optional[Path] = None,
    ) -> "DataPreprocessor":
        """Load a fitted preprocessor object from disk.

        Parameters
        ----------
        path : Optional[Path]
            Path to the saved preprocessor artifact. If None, uses the
            artifact path defined in model_config.yaml.
        features_config_path : Optional[Path]
            Path to features.yaml configuration file.
        model_config_path : Optional[Path]
            Path to model_config.yaml configuration file.

        Returns
        -------
        DataPreprocessor
            A fully fitted DataPreprocessor instance ready for transformation.
        """
        instance = cls(features_config_path, model_config_path)

        if path is None:
            relative_path = instance.model_config["artifacts"]["scaler_path"]
            path = PROJECT_ROOT / relative_path
        else:
            path = Path(path)

        artifact = joblib.load(path)
        instance.scaler = artifact["scaler"]
        
        # Verify feature names match config
        if instance.feature_names != artifact["feature_names"]:
            raise ValueError(
                f"Feature names in loaded artifact {artifact['feature_names']} "
                f"do not match current config {instance.feature_names}."
            )
            
        instance.is_fitted = True
        return instance
