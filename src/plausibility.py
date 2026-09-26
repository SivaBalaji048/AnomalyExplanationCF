"""Plausibility validation module for counterfactual generation.

This module answers the question: "Does this counterfactual state resemble
the learned normal operating population?" using Mahalanobis distance.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


class PlausibilityChecker:
    """Evaluates the plausibility of a counterfactual state.
    
    Uses Mahalanobis distance in the standardized detector-feature space
    relative to the empirical normal training distribution.
    """

    def __init__(
        self,
        normal_training_data: pd.DataFrame,
        preprocessor,
        config_path: str = "config/features.yaml"
    ):
        """Initialize the PlausibilityChecker and build the reference distribution.
        
        Parameters
        ----------
        normal_training_data : pd.DataFrame
            The normal training partition containing original, unscaled values.
        preprocessor : DataPreprocessor
            A fitted preprocessor instance from the project.
        config_path : str
            Path to the features configuration file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        try:
            self.detector_features = self.config["feature_sets"]["detector_inputs"]
        except KeyError:
            raise ValueError("Configuration missing feature_sets.detector_inputs")
            
        if not self.detector_features:
            raise ValueError("Configured detector_inputs list is empty.")

        # Validate preprocessor
        if not hasattr(preprocessor, "is_fitted") or not preprocessor.is_fitted:
            raise ValueError("Provided preprocessor is not fitted.")
            
        self.preprocessor = preprocessor
        
        # Validate training data
        if not isinstance(normal_training_data, pd.DataFrame):
            raise ValueError("normal_training_data must be a pandas DataFrame.")
        if len(normal_training_data) == 0:
            raise ValueError("normal_training_data must contain at least one row.")
            
        if list(normal_training_data.columns) != self.detector_features:
            raise ValueError(
                f"normal_training_data columns must exactly match configured detector features in order.\n"
                f"Expected: {self.detector_features}\n"
                f"Actual: {list(normal_training_data.columns)}"
            )
            
        # Extract features in correct order
        features_df = normal_training_data[self.detector_features]
        
        # Check for numeric, non-bool, and finite types
        for col in self.detector_features:
            if not pd.api.types.is_numeric_dtype(features_df[col]):
                raise ValueError(f"Feature {col} in normal_training_data must be numeric.")
            if pd.api.types.is_bool_dtype(features_df[col]):
                raise ValueError(f"Feature {col} in normal_training_data must not be boolean.")
                
        if not np.isfinite(features_df.values).all():
            raise ValueError("normal_training_data contains NaN or infinite values.")
            
        # Apply standard transformation
        self.transformed_training_data = self.preprocessor.transform(features_df)
        
        # Handle the preprocessor's actual return type
        if isinstance(self.transformed_training_data, pd.DataFrame):
            matrix = self.transformed_training_data.values
        else:
            matrix = np.asarray(self.transformed_training_data)
            
        # Validate transformed training dimension
        expected_dim = len(self.detector_features)
        actual_dim = matrix.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Transformed training data has incorrect dimensions.\n"
                f"Expected features: {expected_dim}\n"
                f"Actual features: {actual_dim}"
            )
            
        # Calculate statistics
        self.mean_vector = np.mean(matrix, axis=0)
        
        # Calculate the covariance matrix (rowvar=False means columns are variables)
        self.covariance_matrix = np.cov(matrix, rowvar=False)
        
        # Validate dimensions
        expected_dim = len(self.detector_features)
        if self.covariance_matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Covariance matrix has incorrect dimensions: {self.covariance_matrix.shape}. "
                f"Expected: {(expected_dim, expected_dim)}."
            )
            
        # Calculate pseudo-inverse for numerical stability
        self.inv_covariance_matrix = np.linalg.pinv(self.covariance_matrix)
        
        # Calculate distance for all records
        self.training_population_size = len(normal_training_data)
        training_distances = []
        for i in range(self.training_population_size):
            dist = self._calculate_mahalanobis(matrix[i])
            training_distances.append(dist)
            
        # Determine 95th percentile threshold
        self.threshold = float(np.percentile(training_distances, 95))
        
    def _calculate_mahalanobis(self, x: np.ndarray) -> float:
        """Calculate the Mahalanobis distance for a standardized vector x."""
        diff = x - self.mean_vector
        distance_sq = np.dot(np.dot(diff, self.inv_covariance_matrix), diff)
        
        # Clamp tiny numerical noise
        if -1e-10 < distance_sq < 0:
            distance_sq = 0.0
            
        if distance_sq < 0:
            raise ValueError(f"Calculated squared Mahalanobis distance is negative: {distance_sq}")
            
        if not np.isfinite(distance_sq):
            raise ValueError("Calculated Mahalanobis distance is non-finite.")
            
        return float(np.sqrt(distance_sq))

    def check(self, counterfactual_state: pd.DataFrame) -> dict:
        """Evaluate whether the candidate state is plausible.
        
        Parameters
        ----------
        counterfactual_state : pd.DataFrame
            The proposed counterfactual state containing original, unscaled values.
            
        Returns
        -------
        dict
            A structured result detailing the plausibility decision.
        """
        # Validate counterfactual state
        if not isinstance(counterfactual_state, pd.DataFrame):
            raise ValueError("counterfactual_state must be a pandas DataFrame.")
            
        if len(counterfactual_state) != 1:
            raise ValueError("counterfactual_state must contain exactly one row.")
            
        if list(counterfactual_state.columns) != self.detector_features:
            raise ValueError(
                f"counterfactual_state columns must exactly match configured detector features in order.\n"
                f"Expected: {self.detector_features}\n"
                f"Actual: {list(counterfactual_state.columns)}"
            )
            
        cf_features = counterfactual_state[self.detector_features]
        
        for col in self.detector_features:
            if not pd.api.types.is_numeric_dtype(cf_features[col]):
                raise ValueError(f"Feature {col} must be numeric in counterfactual_state.")
            if pd.api.types.is_bool_dtype(cf_features[col]):
                raise ValueError(f"Feature {col} must not be boolean in counterfactual_state.")
                
        if not np.isfinite(cf_features.values).all():
            raise ValueError("counterfactual_state contains NaN or infinite values.")
            
        # Standardize the candidate
        transformed_cf = self.preprocessor.transform(cf_features)
        
        # Handle the preprocessor's actual return type
        if isinstance(transformed_cf, pd.DataFrame):
            cf_matrix = transformed_cf.values
        else:
            cf_matrix = np.asarray(transformed_cf)
            
        # Validate transformed counterfactual dimension
        expected_dim = len(self.detector_features)
        actual_dim = cf_matrix.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Transformed counterfactual data has incorrect dimensions.\n"
                f"Expected features: {expected_dim}\n"
                f"Actual features: {actual_dim}"
            )
            
        # Calculate distance
        x = cf_matrix[0]
        distance = self._calculate_mahalanobis(x)
        
        # Threshold comparison
        is_plausible = bool(distance <= self.threshold)
        
        reason = (
            "Counterfactual is within the derived normal-distribution plausibility threshold."
            if is_plausible
            else "Counterfactual is outside the derived normal-distribution plausibility threshold."
        )
        
        return {
            "plausible": is_plausible,
            "distance": distance,
            "threshold": self.threshold,
            "method": "mahalanobis",
            "reference_population_size": self.training_population_size,
            "reference_percentile": 95,
            "reason": reason
        }
