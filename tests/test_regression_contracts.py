"""
Regression and contract tests for Automated Anomaly Explanation with Counterfactuals.

Validates that the frozen baseline, policies, configurations, and generated
artifacts remain intact and reproducible.
"""

import json
import math
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import yaml

from src.preprocessing import DataPreprocessor
from src.detector import AnomalyDetector
from src.feasibility import FeasibilityChecker
from src.plausibility import PlausibilityChecker
from src.counterfactual import CounterfactualEngine
from src.pipeline import ExplanationPipeline

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = REPO_ROOT / "data/raw/ai4i2020.csv"
FEATURES_CONFIG_PATH = REPO_ROOT / "config/features.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "config/model_config.yaml"

SCALER_PATH = REPO_ROOT / "models/scaler.joblib"
MODEL_PATH = REPO_ROOT / "models/isolation_forest.joblib"

DETECTOR_METRICS_PATH = REPO_ROOT / "results/metrics/detector_metrics.json"
CF_METRICS_PATH = REPO_ROOT / "results/metrics/counterfactual_metrics.json"
CF_RESULTS_CSV_PATH = REPO_ROOT / "results/explanations/counterfactual_case_results.csv"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EXPECTED_DETECTOR_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

EXPECTED_MUTABLE_FEATURES = [
    "Rotational speed [rpm]",
    "Torque [Nm]",
]

EXPECTED_IMMUTABLE_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Tool wear [min]",
]


# -----------------------------------------------------------------------------
# Shared Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def features_config():
    with open(FEATURES_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def model_config():
    with open(MODEL_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv(DATA_PATH)

@pytest.fixture(scope="module")
def split_datasets(raw_data):
    preprocessor = DataPreprocessor(
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )
    return preprocessor.prepare_datasets(raw_data)

@pytest.fixture(scope="module")
def frozen_scaler():
    return DataPreprocessor.load(
        path=SCALER_PATH,
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )

@pytest.fixture(scope="module")
def frozen_detector():
    return AnomalyDetector.load(
        path=MODEL_PATH,
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )

@pytest.fixture(scope="module")
def evaluation_df(split_datasets):
    _, normal_eval, failure_eval = split_datasets
    return pd.concat([normal_eval, failure_eval], ignore_index=True)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_configuration_contract(features_config, model_config):
    """1. Configuration contract test"""
    
    # Feature order
    assert features_config["feature_sets"]["detector_inputs"] == EXPECTED_DETECTOR_FEATURES
    
    # Mutable/Immutable policies
    assert features_config["counterfactual_policy"]["mutable_features"] == EXPECTED_MUTABLE_FEATURES
    
    policy_immutable = features_config["counterfactual_policy"]["immutable_features"]
    for f in EXPECTED_IMMUTABLE_FEATURES:
        assert f in policy_immutable
        
    # Excluded metadata/identifiers
    assert "Machine failure" in policy_immutable
    assert "UDI" in policy_immutable
    assert "Type" in policy_immutable
    
    for feature_name, feature_data in features_config["features"].items():
        if feature_name in EXPECTED_DETECTOR_FEATURES:
            assert feature_data["role"] == "detector_input"
        else:
            assert feature_data["role"] != "detector_input"
            
    # Bounds source
    assert features_config["counterfactual_policy"]["bounds_source"] == "observed_dataset_values"
    assert features_config["counterfactual_policy"]["bounds_are_certified_safety_limits"] is False
    
    # Search limits
    assert model_config["counterfactual_search"]["max_candidates"] == 5000
    assert model_config["counterfactual_search"]["max_generation_time_seconds"] == 10


def test_frozen_artifact_loading_and_feature_contract(split_datasets, frozen_scaler, frozen_detector):
    """2. Frozen artifact loading and feature-contract test"""
    
    assert frozen_scaler.is_fitted is True
    assert frozen_detector.is_fitted is True
    
    assert frozen_scaler.feature_names == EXPECTED_DETECTOR_FEATURES
    assert frozen_detector.feature_names == EXPECTED_DETECTOR_FEATURES
    
    _, normal_eval, _ = split_datasets
    single_record = normal_eval.iloc[[0]][EXPECTED_DETECTOR_FEATURES]
    
    transformed_record = frozen_scaler.transform(single_record)
    
    # Assert transformed output contains exactly the five approved feature columns in order
    assert list(transformed_record.columns) == EXPECTED_DETECTOR_FEATURES
    
    # Assert detector prediction is either -1 or 1, and its decision-function score is finite
    pred = frozen_detector.predict_labels(transformed_record).iloc[0]
    assert pred in [-1, 1]
    
    score = frozen_detector.predict_anomaly_score(transformed_record).iloc[0]
    assert math.isfinite(score)


def test_frozen_detector_baseline_regression(evaluation_df, frozen_scaler, frozen_detector):
    """3. Frozen detector baseline regression test"""
    
    X_eval = evaluation_df[EXPECTED_DETECTOR_FEATURES]
    X_scaled = frozen_scaler.transform(X_eval)
    predicted_labels = frozen_detector.predict_labels(X_scaled)
    anomaly_mask = predicted_labels == -1
    
    y_true = evaluation_df["Machine failure"].values
    y_pred = anomaly_mask.astype(int)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    
    anomalies_df = evaluation_df[anomaly_mask]
    known_failures_detected = (anomalies_df["Machine failure"] == 1).sum()
    known_normals_detected = (anomalies_df["Machine failure"] == 0).sum()
    
    error_msg = "Mismatch means the frozen baseline, data split, feature contract, or artifact may have changed."
    
    assert len(evaluation_df) == 2272, error_msg
    assert tn == 1905, error_msg
    assert fp == 28, error_msg
    assert fn == 300, error_msg
    assert tp == 39, error_msg
    assert len(anomalies_df) == 67, error_msg
    assert known_failures_detected == 39, error_msg
    assert known_normals_detected == 28, error_msg


def test_feasibility_policy_regression(split_datasets, features_config):
    """4. Feasibility policy regression test"""
    
    normal_train, _, _ = split_datasets
    original_state = normal_train.iloc[[0]][EXPECTED_DETECTOR_FEATURES].copy()
    
    checker = FeasibilityChecker(config_path=FEATURES_CONFIG_PATH)
    
    # Mutable candidate
    allowed_candidate = original_state.copy()
    torque_step = features_config["features"]["Torque [Nm]"]["counterfactual_step"]
    torque_max = features_config["features"]["Torque [Nm]"]["max"]
    
    # Make sure we don't exceed empirical bound
    new_torque = allowed_candidate["Torque [Nm]"].iloc[0] + torque_step
    if new_torque > torque_max:
        new_torque = allowed_candidate["Torque [Nm]"].iloc[0] - torque_step
    allowed_candidate["Torque [Nm]"] = new_torque
    
    # Assert feasible
    allowed_result = checker.check(original_state, allowed_candidate)
    assert allowed_result["feasible"] is True
    
    # Immutable candidate
    prohibited_candidate = original_state.copy()
    prohibited_candidate["Air temperature [K]"] += 1.0
    
    # Assert infeasible
    prohibited_result = checker.check(original_state, prohibited_candidate)
    assert prohibited_result["feasible"] is False
    assert prohibited_result["checks"]["immutable_features_unchanged"] is False


def test_plausibility_policy_regression(split_datasets, frozen_scaler):
    """5. Plausibility policy regression test"""
    
    normal_train, _, _ = split_datasets
    normal_training_features = normal_train[EXPECTED_DETECTOR_FEATURES].copy()
    
    checker = PlausibilityChecker(
        normal_training_data=normal_training_features,
        preprocessor=frozen_scaler,
        config_path=FEATURES_CONFIG_PATH
    )
    
    mean_state = normal_training_features.mean().to_frame().T
    
    mean_result = checker.check(mean_state)
    assert mean_result["method"] == "mahalanobis"
    assert math.isfinite(mean_result["threshold"])
    assert mean_result["threshold"] > 0
    assert mean_result["reference_percentile"] == 95
    assert mean_result["plausible"] is True


def test_end_to_end_counterfactual_pipeline_regression(evaluation_df, frozen_scaler, frozen_detector, split_datasets):
    """6. One end-to-end counterfactual pipeline regression test"""
    
    normal_train, _, _ = split_datasets
    
    # Known fixed evaluation row index 1933
    sample = evaluation_df.iloc[[1933]][EXPECTED_DETECTOR_FEATURES].astype(float)
    
    # Assert detector still predicts this as anomalous
    assert frozen_detector.predict_labels(frozen_scaler.transform(sample)).iloc[0] == -1
    
    # Build pipeline
    plausibility_checker = PlausibilityChecker(
        normal_training_data=normal_train[EXPECTED_DETECTOR_FEATURES].copy(),
        preprocessor=frozen_scaler,
        config_path=FEATURES_CONFIG_PATH,
    )
    engine = CounterfactualEngine(
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )
    feasibility_checker = FeasibilityChecker(config_path=FEATURES_CONFIG_PATH)
    
    pipeline = ExplanationPipeline(
        preprocessor=frozen_scaler,
        detector=frozen_detector,
        engine=engine,
        feasibility_checker=feasibility_checker,
        plausibility_checker=plausibility_checker,
    )
    
    result = pipeline.analyze(sample)
    
    assert result["found"] is True
    assert result["counterfactual_label"] == 1
    assert result["feasibility_status"] == "feasible"
    assert result["candidates_evaluated"] <= 5000
    assert math.isfinite(result["plausibility_distance"])
    assert math.isfinite(result["plausibility_threshold"])
    
    # Assert changed features are only mutable ones
    for changed in result["changed_features"]:
        assert changed["feature"] in EXPECTED_MUTABLE_FEATURES


def test_generated_results_schema_regression():
    """7. Generated-results schema regression test"""
    
    missing_files_msg = (
        "Generated results files are missing. "
        "Please first run:\n"
        ".\\.venv\\Scripts\\python.exe -m tests.manual_generate_results"
    )
    
    assert DETECTOR_METRICS_PATH.exists(), missing_files_msg
    assert CF_METRICS_PATH.exists(), missing_files_msg
    assert CF_RESULTS_CSV_PATH.exists(), missing_files_msg
    
    # Detector JSON
    with open(DETECTOR_METRICS_PATH, "r") as f:
        detector_metrics = json.load(f)
        
    assert detector_metrics["evaluation_record_count"] == 2272
    assert detector_metrics["performance"]["confusion_matrix"]["TP"] == 39
    
    # Counterfactual JSON
    with open(CF_METRICS_PATH, "r") as f:
        cf_metrics = json.load(f)
        
    assert cf_metrics["primary_counterfactual_evaluation_cases"] == 39
    assert cf_metrics["counterfactuals_found"] == 36
    assert cf_metrics["counterfactuals_not_found"] == 3
    assert cf_metrics["plausible_counterfactuals"] == 9
    assert cf_metrics["plausibility_threshold"] > 0
    
    # CSV Data
    df = pd.read_csv(CF_RESULTS_CSV_PATH)
    assert len(df) == 39
    
    expected_columns = [
        "evaluation_row_index",
        "true_machine_failure",
        "detector_prediction",
        "detector_anomaly_score",
        "counterfactual_found",
        "failure_reason",
        "candidates_evaluated",
        "generation_time_seconds",
        "changed_features",
        "sparsity",
        "normalized_proximity",
        "feasibility_status",
        "plausibility_status",
        "plausibility_distance",
        "plausibility_threshold",
    ]
    for col in expected_columns:
        assert col in df.columns
        
    found_count = df["counterfactual_found"].sum()
    not_found_count = (~df["counterfactual_found"]).sum()
    
    assert found_count == 36
    assert not_found_count == 3
    
    not_found_rows = df[~df["counterfactual_found"]]
    assert list(not_found_rows["evaluation_row_index"].astype(int)) == [1954, 1955, 2166]
    
    for _, row in not_found_rows.iterrows():
        for feature in EXPECTED_DETECTOR_FEATURES:
            cf_val = row[f"counterfactual_{feature}"]
            assert pd.isna(cf_val) or str(cf_val).strip() == ""
