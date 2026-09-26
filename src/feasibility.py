import math
import yaml
import numpy as np
import pandas as pd

class FeasibilityChecker:
    def __init__(self, config_path="config/features.yaml"):
        """
        Initializes the FeasibilityChecker by loading feasibility rules from the configuration file.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.detector_features = self.config.get("feature_sets", {}).get("detector_inputs", [])
        if not self.detector_features:
            raise ValueError("No detector_inputs found in configuration.")
            
        self.mutable_features = self.config.get("counterfactual_policy", {}).get("mutable_features", [])
        
        # Calculate immutable detector features
        self.immutable_detector_features = [
            f for f in self.detector_features if f not in self.mutable_features
        ]
        
        self.feature_details = self.config.get("features", {})
        
        # Configuration Validation for mutable features
        for feat in self.mutable_features:
            if feat not in self.feature_details:
                raise ValueError(f"Mutable feature '{feat}' is missing from features configuration.")
            
            feat_config = self.feature_details[feat]
            
            if "min" not in feat_config or "max" not in feat_config:
                raise ValueError(f"Mutable feature '{feat}' must have configured min and max bounds.")
                
            min_val = feat_config["min"]
            max_val = feat_config["max"]
            
            if not isinstance(min_val, (int, float, np.number)) or not isinstance(max_val, (int, float, np.number)):
                raise ValueError(f"Mutable feature '{feat}' min and max must be numeric.")
                
            if min_val > max_val:
                raise ValueError(f"Mutable feature '{feat}' min ({min_val}) cannot be greater than max ({max_val}).")
                
            if "counterfactual_step" not in feat_config:
                raise ValueError(f"Mutable feature '{feat}' must have a configured counterfactual_step.")
                
            step = feat_config["counterfactual_step"]
            if not isinstance(step, (int, float, np.number)) or step <= 0:
                raise ValueError(f"Mutable feature '{feat}' counterfactual_step must be a positive number.")
        
    def _validate_inputs(self, original_state: pd.DataFrame, counterfactual_state: pd.DataFrame):
        """
        Validates the structure and content of the input DataFrames.
        """
        if not isinstance(original_state, pd.DataFrame):
            raise ValueError("original_state must be a pandas DataFrame")
        if not isinstance(counterfactual_state, pd.DataFrame):
            raise ValueError("counterfactual_state must be a pandas DataFrame")
            
        if len(original_state) != 1:
            raise ValueError(f"original_state must contain exactly one row, got {len(original_state)}")
        if len(counterfactual_state) != 1:
            raise ValueError(f"counterfactual_state must contain exactly one row, got {len(counterfactual_state)}")
            
        orig_cols = list(original_state.columns)
        cf_cols = list(counterfactual_state.columns)
        
        if orig_cols != self.detector_features:
            raise ValueError(f"original_state columns {orig_cols} do not match configured detector features {self.detector_features}")
            
        if cf_cols != self.detector_features:
            raise ValueError(f"counterfactual_state columns {cf_cols} do not match configured detector features {self.detector_features}")
            
        for df, name in [(original_state, "original_state"), (counterfactual_state, "counterfactual_state")]:
            for col in self.detector_features:
                val = df.iloc[0][col]
                if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
                    raise ValueError(f"{name} feature '{col}' must be numeric, got {type(val)}")
                if pd.isna(val) or not np.isfinite(val):
                    raise ValueError(f"{name} feature '{col}' must be a finite numeric value, got {val}")

    def check(self, original_state: pd.DataFrame, counterfactual_state: pd.DataFrame) -> dict:
        """
        Validates whether the proposed counterfactual_state satisfies the project's
        configured feasibility/actionability rules relative to the original_state.
        """
        self._validate_inputs(original_state, counterfactual_state)
        
        orig_row = original_state.iloc[0]
        cf_row = counterfactual_state.iloc[0]
        
        changed_features = []
        for feature in self.detector_features:
            orig_val = float(orig_row[feature])
            cf_val = float(cf_row[feature])
            
            # Using a very small tolerance to detect actual intended changes vs floating-point noise
            if not math.isclose(orig_val, cf_val, rel_tol=0.0, abs_tol=1e-12):
                changed_features.append({
                    "feature": feature,
                    "original_value": orig_val,
                    "counterfactual_value": cf_val,
                    "delta": cf_val - orig_val
                })
                
        mutable_features_only = True
        immutable_features_unchanged = True
        bounds_valid = True
        step_alignment_valid = True
        violations = []
        
        # Check #1: Mutable features only & Check #2: Immutable features unchanged
        for change in changed_features:
            feat = change["feature"]
            if feat not in self.mutable_features:
                mutable_features_only = False
                immutable_features_unchanged = False
                violations.append(f"Feature '{feat}' is immutable but was changed from {change['original_value']} to {change['counterfactual_value']}.")
                
        # Check #3: Bounds & Check #4: Step alignment (for changed mutable features)
        for change in changed_features:
            feat = change["feature"]
            if feat in self.mutable_features:
                cf_val = change["counterfactual_value"]
                orig_val = change["original_value"]
                feat_config = self.feature_details.get(feat, {})
                
                if "min" in feat_config and "max" in feat_config:
                    min_val = feat_config["min"]
                    max_val = feat_config["max"]
                    if cf_val < min_val or cf_val > max_val:
                        bounds_valid = False
                        violations.append(f"Feature '{feat}' counterfactual value {cf_val} is outside configured bounds [{min_val}, {max_val}].")
                        
                if "counterfactual_step" in feat_config:
                    step = feat_config["counterfactual_step"]
                    step_count = (cf_val - orig_val) / step
                    change["step_count"] = step_count
                    
                    if not math.isclose(step_count, round(step_count), rel_tol=0.0, abs_tol=1e-9):
                        step_alignment_valid = False
                        violations.append(f"Feature '{feat}' change of {cf_val - orig_val} is not aligned with configured step {step}.")

        feasible = mutable_features_only and immutable_features_unchanged and bounds_valid and step_alignment_valid
        
        if feasible:
            reason = "All configured feasibility constraints satisfied."
        else:
            reason = "One or more configured feasibility constraints were violated."
            
        return {
            "feasible": feasible,
            "checks": {
                "mutable_features_only": mutable_features_only,
                "immutable_features_unchanged": immutable_features_unchanged,
                "bounds_valid": bounds_valid,
                "step_alignment_valid": step_alignment_valid
            },
            "changed_features": changed_features,
            "violations": violations,
            "reason": reason
        }
