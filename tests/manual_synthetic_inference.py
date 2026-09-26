"""
Manual synthetic inference test runner.

Purpose
-------
Evaluate the frozen anomaly-detection and explanation pipeline on one
synthetic dataset at a time.

Workflow
--------
1. Load the frozen preprocessor.
2. Load the frozen Isolation Forest detector.
3. Load one synthetic CSV.
4. Run anomaly detection on ALL rows.
5. Select up to 5 deepest anomalies using the lowest decision-function
   scores.
6. Run the expensive counterfactual + feasibility + plausibility pipeline
   only on those selected anomalies.
7. Print a dataset-level summary.

Important
---------
- No retraining is performed.
- No scaler refitting is performed.
- Synthetic ground-truth labels are not required.
- Synthetic labels are not used for anomaly detection.
- The original AI4I dataset is used only to reconstruct the normal
  training population required by the existing plausibility checker.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_raw_dataset
from src.preprocessing import DataPreprocessor
from src.detector import AnomalyDetector
from src.counterfactual import CounterfactualEngine
from src.feasibility import FeasibilityChecker
from src.plausibility import PlausibilityChecker
from src.pipeline import ExplanationPipeline


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai4i2020.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest.joblib"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "scaler.joblib"
)

FEATURES_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "features.yaml"
)

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config.yaml"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DETECTOR_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Expensive counterfactual analysis is performed only on this many anomalies.
MAX_DETAILED_CASES = 5


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def print_section(title: str) -> None:
    """Print a consistent section heading."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def validate_test_dataset(df: pd.DataFrame) -> None:
    """
    Validate that the synthetic dataset contains the exact detector
    features required by the frozen model.

    Synthetic datasets do not need Machine failure/TWF/HDF/PWF/OSF/RNF
    labels for this inference test.
    """

    missing_features = [
        feature
        for feature in DETECTOR_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Synthetic dataset is missing required detector features: "
            + ", ".join(missing_features)
        )

    detector_data = df[DETECTOR_FEATURES]

    if detector_data.isnull().any().any():
        missing_columns = detector_data.columns[
            detector_data.isnull().any()
        ].tolist()

        raise ValueError(
            "Unexpected missing values found in detector features: "
            + ", ".join(missing_columns)
        )

    for feature in DETECTOR_FEATURES:
        if not pd.api.types.is_numeric_dtype(
            detector_data[feature]
        ):
            raise TypeError(
                f"Detector feature '{feature}' must be numeric."
            )


def select_representative_anomalies(
    scores: np.ndarray,
    labels: np.ndarray,
    max_cases: int = 5,
) -> list[int]:
    """
    Select representative anomalous records across the anomaly-score range.

    Instead of selecting only the deepest anomalies, this selects anomalies
    at approximately 0%, 25%, 50%, 75%, and 100% of the anomaly-score range.

    This gives us a mixture of:
        - very deep anomalies
        - moderately deep anomalies
        - middle-range anomalies
        - near-boundary anomalies

    Returns:
        List of original row indices.
    """

    anomaly_indices = np.flatnonzero(labels == -1)

    if len(anomaly_indices) == 0:
        return []

    # Sort anomalous records from most anomalous to least anomalous.
    sorted_indices = anomaly_indices[
        np.argsort(scores[anomaly_indices])
    ]

    if len(sorted_indices) <= max_cases:
        return sorted_indices.tolist()

    # Cover the full anomaly-score distribution.
    quantiles = np.linspace(0.0, 1.0, max_cases)

    selected_positions = [
        int(round(q * (len(sorted_indices) - 1)))
        for q in quantiles
    ]

    # Remove any accidental duplicates while preserving order.
    selected = []
    seen = set()

    for position in selected_positions:
        index = int(sorted_indices[position])

        if index not in seen:
            selected.append(index)
            seen.add(index)

    # Safety fallback: fill remaining slots if duplicates occurred.
    if len(selected) < max_cases:
        for index in sorted_indices:
            index = int(index)

            if index not in seen:
                selected.append(index)
                seen.add(index)

            if len(selected) == max_cases:
                break

    return selected


def get_result_value(
    result: dict,
    *keys,
):
    """
    Return the first available value from a result dictionary.
    """

    for key in keys:
        if key in result:
            return result[key]

    return None


def print_feature_changes(
    original_state: pd.DataFrame,
    counterfactual_state,
) -> None:
    """
    Print original -> counterfactual feature changes when the pipeline
    returns a counterfactual state.
    """

    if counterfactual_state is None:
        return

    try:
        original_row = (
            original_state.iloc[0]
        )

        if isinstance(
            counterfactual_state,
            pd.DataFrame,
        ):
            cf_row = (
                counterfactual_state.iloc[0]
            )

        elif isinstance(
            counterfactual_state,
            pd.Series,
        ):
            cf_row = counterfactual_state

        elif isinstance(
            counterfactual_state,
            dict,
        ):
            cf_row = pd.Series(
                counterfactual_state
            )

        else:
            return

        changes_found = False

        print()
        print("Feature changes:")

        for feature in DETECTOR_FEATURES:

            if feature not in cf_row.index:
                continue

            original_value = (
                original_row[feature]
            )

            new_value = cf_row[feature]

            try:
                changed = not np.isclose(
                    float(original_value),
                    float(new_value),
                )
            except (
                TypeError,
                ValueError,
            ):
                changed = (
                    original_value
                    != new_value
                )

            if changed:

                changes_found = True

                print(
                    f"  {feature}"
                )

                print(
                    f"    Original       : "
                    f"{original_value}"
                )

                print(
                    f"    Counterfactual : "
                    f"{new_value}"
                )

                try:
                    delta = (
                        float(new_value)
                        - float(original_value)
                    )

                    print(
                        f"    Delta          : "
                        f"{delta}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        if not changes_found:
            print(
                "  No feature changes detected."
            )

    except Exception as exc:
        print(
            "Could not display feature changes: "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    # ========================================================================
    # COMMAND-LINE ARGUMENT
    # ========================================================================

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python -m tests.manual_synthetic_inference "
            "<synthetic_csv_path>\n\n"
            "Example:\n"
            "  python -m tests.manual_synthetic_inference "
            "data/raw/Synthetic_Test_1_clean.csv"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    if not input_path.is_absolute():
        input_path = (
            PROJECT_ROOT
            / input_path
        )

    input_path = input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found:\n"
            f"{input_path}"
        )

    # ========================================================================
    # HEADER
    # ========================================================================

    print_section(
        "SYNTHETIC DATASET INFERENCE TEST"
    )

    print(
        f"Dataset     : {input_path.name}"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print()
    print(
        "Mode: FROZEN MODEL INFERENCE / EVALUATION ONLY"
    )

    print()
    print(
        "No retraining or scaler refitting will be performed."
    )

    # ========================================================================
    # LOAD FROZEN ARTIFACTS
    # ========================================================================

    print_section(
        "LOADING FROZEN ARTIFACTS"
    )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler artifact not found:\n"
            f"{SCALER_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Detector artifact not found:\n"
            f"{MODEL_PATH}"
        )

    load_start = time.perf_counter()

    preprocessor = (
        DataPreprocessor.load(
            SCALER_PATH
        )
    )

    detector = (
        AnomalyDetector.load(
            MODEL_PATH
        )
    )

    load_time = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Scaler : {SCALER_PATH}"
    )

    print(
        f"Model  : {MODEL_PATH}"
    )

    print(
        f"Load time: {load_time:.4f} sec"
    )

    if not getattr(
        preprocessor,
        "is_fitted",
        True,
    ):
        raise RuntimeError(
            "Loaded preprocessor is not fitted."
        )

    if not detector.is_fitted:
        raise RuntimeError(
            "Loaded detector is not fitted."
        )

    # ========================================================================
    # LOAD SYNTHETIC DATASET
    # ========================================================================

    print_section(
        "LOADING SYNTHETIC DATASET"
    )

    dataset_start = time.perf_counter()

    # Synthetic inference datasets intentionally do not contain
    # ground-truth failure labels.
    #
    # Therefore, do NOT use load_raw_dataset() here because that loader
    # performs strict AI4I validation and requires all label columns.
    #
    # The frozen model only requires the five detector input features.
    test_df = pd.read_csv(
        input_path
    )

    dataset_load_time = (
        time.perf_counter()
        - dataset_start
    )

    print(
        f"Rows    : {len(test_df)}"
    )

    print(
        f"Columns : {len(test_df.columns)}"
    )

    print(
        f"Load time: "
        f"{dataset_load_time:.4f} sec"
    )

    validate_test_dataset(
        test_df
    )

    print()
    print(
        "Detector features:"
    )

    for feature in DETECTOR_FEATURES:
        print(
            f"  - {feature}"
        )

    # ========================================================================
    # FULL DATASET DETECTION
    # ========================================================================

    print_section(
        "FULL-DATASET DETECTOR RESULTS"
    )

    detector_input = (
        test_df[
            DETECTOR_FEATURES
        ].copy()
    )

    # ------------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------------

    transform_start = (
        time.perf_counter()
    )

    X_scaled = (
        preprocessor.transform(
            detector_input
        )
    )

    transform_time = (
        time.perf_counter()
        - transform_start
    )

    # ------------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------------

    prediction_start = (
        time.perf_counter()
    )

    predicted_labels = (
        detector.predict_labels(
            X_scaled
        )
    )

    anomaly_scores = (
        detector.predict_anomaly_score(
            X_scaled
        )
    )

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    predicted_labels = np.asarray(
        predicted_labels
    )

    anomaly_scores = np.asarray(
        anomaly_scores
    )

    # ------------------------------------------------------------------------
    # Validate output lengths
    # ------------------------------------------------------------------------

    if len(predicted_labels) != len(
        test_df
    ):
        raise RuntimeError(
            "Detector returned a different number of labels "
            "than input rows."
        )

    if len(anomaly_scores) != len(
        test_df
    ):
        raise RuntimeError(
            "Detector returned a different number of scores "
            "than input rows."
        )

    # ------------------------------------------------------------------------
    # Detector statistics
    # ------------------------------------------------------------------------

    anomaly_mask = (
        predicted_labels == -1
    )

    normal_mask = (
        predicted_labels == 1
    )

    total_records = len(
        test_df
    )

    normal_count = int(
        np.sum(normal_mask)
    )

    anomaly_count = int(
        np.sum(anomaly_mask)
    )

    anomaly_rate = (
        anomaly_count
        / total_records
        * 100
        if total_records > 0
        else 0.0
    )

    total_detection_time = (
        transform_time
        + prediction_time
    )

    print(
        f"Total records : {total_records}"
    )

    print(
        f"Normal        : {normal_count}"
    )

    print(
        f"Anomalous     : {anomaly_count}"
    )

    print(
        f"Anomaly rate  : "
        f"{anomaly_rate:.2f}%"
    )

    print()

    print(
        f"Transform time : "
        f"{transform_time:.4f} sec"
    )

    print(
        f"Prediction time: "
        f"{prediction_time:.4f} sec"
    )

    print(
        f"Total detection: "
        f"{total_detection_time:.4f} sec"
    )

    if total_records > 0:

        print(
            "Avg detection/record: "
            f"{total_detection_time / total_records:.6f} sec"
        )

    print()

    print(
        "Score range: "
        f"{np.min(anomaly_scores):.6f} "
        f"to "
        f"{np.max(anomaly_scores):.6f}"
    )

    # ========================================================================
    # NO ANOMALIES
    # ========================================================================

    if anomaly_count == 0:

        print()
        print(
            "No anomalous records were detected."
        )

        print(
            "Counterfactual analysis is not required "
            "for this dataset."
        )

        print_section(
            "FINAL DATASET SUMMARY"
        )

        print(
            f"Dataset           : "
            f"{input_path.name}"
        )

        print(
            f"Total records     : "
            f"{total_records}"
        )

        print(
            f"Normal records    : "
            f"{normal_count}"
        )

        print(
            f"Anomalous records : "
            f"{anomaly_count}"
        )

        print(
            "Detailed CF cases : 0"
        )

        print_section(
            "TEST COMPLETE"
        )

        return

    # ========================================================================
    # SELECT FIVE DEEPEST ANOMALIES
    # ========================================================================

    print_section(
        "REPRESENTATIVE ANOMALY SELECTION"
    )

    selected_rows = (
    select_representative_anomalies(
        scores=anomaly_scores,
        labels=predicted_labels,
        max_cases=MAX_DETAILED_CASES,
    )
)

    print("Selection rule:")

    print("Representative anomalies are selected across the detected "
        "anomaly-score range at approximately 0%, 25%, 50%, 75%, and 100%.")

    print()

    print(
        f"Detected anomalies available: "
        f"{anomaly_count}"
    )

    print(
        f"Detailed cases selected: "
        f"{len(selected_rows)}"
    )

    print()

    for rank, row_index in enumerate(
        selected_rows,
        start=1,
    ):

        print(
            f"{rank}. "
            f"row={row_index} | "
            f"score="
            f"{anomaly_scores[row_index]:.6f}"
        )

    # ========================================================================
    # BUILD PLAUSIBILITY REFERENCE
    # ========================================================================

    print_section(
        "BUILDING PLAUSIBILITY REFERENCE"
    )

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Original AI4I dataset was not found:\n"
            f"{RAW_DATA_PATH}"
        )

    reference_start = (
        time.perf_counter()
    )

    # The original AI4I dataset contains complete labels and therefore
    # can use the project's strict validated loader.
    reference_df = load_raw_dataset(
        RAW_DATA_PATH
    )

    # Use known-normal records only.
    normal_reference_df = (
        reference_df[
            reference_df[
                "Machine failure"
            ] == 0
        ].copy()
    )

    # Reconstruct the exact normal-training split used by the project.
    split_preprocessor = DataPreprocessor(
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )

    normal_training_df, normal_eval_df, failure_eval_df = split_preprocessor.prepare_datasets(
        reference_df
    )

    # IMPORTANT:
    # PlausibilityChecker explicitly requires ORIGINAL, UNSCALED
    # detector-feature values as a pandas DataFrame.
    #
    # Do NOT call preprocessor.transform() on this DataFrame.
    #
    # PlausibilityChecker receives the original values AND the fitted
    # preprocessor and performs its own standardized-space processing.
    normal_training_features = (
        normal_training_df[
            DETECTOR_FEATURES
        ].copy()
    )

    plausibility_checker = (
        PlausibilityChecker(
            normal_training_data=normal_training_features,
            preprocessor=preprocessor,
            config_path=str(
                FEATURES_CONFIG_PATH
            ),
        )
    )

    reference_time = (
        time.perf_counter()
        - reference_start
    )

    print(
        f"Normal reference records : "
        f"{len(normal_reference_df)}"
    )

    print(
        f"Normal training records  : "
        f"{len(normal_training_df)}"
    )

    print(
        f"Reference build time     : "
        f"{reference_time:.4f} sec"
    )

    # ========================================================================
    # INITIALIZE PIPELINE
    # ========================================================================

    print_section(
        "INITIALIZING EXPLANATION PIPELINE"
    )

    # CounterfactualEngine uses both project configuration files.
    counterfactual_engine = (
        CounterfactualEngine(
            features_config_path=str(
                FEATURES_CONFIG_PATH
            ),
            model_config_path=str(
                MODEL_CONFIG_PATH
            ),
        )
    )

    # Actual FeasibilityChecker constructor:
    #
    # FeasibilityChecker(
    #     config_path="config/features.yaml"
    # )
    feasibility_checker = (
        FeasibilityChecker(
            config_path=str(
                FEATURES_CONFIG_PATH
            )
        )
    )

    # Actual ExplanationPipeline constructor:
    #
    # ExplanationPipeline(
    #     preprocessor,
    #     detector,
    #     engine,
    #     feasibility_checker,
    #     plausibility_checker
    # )
    #
    # It does NOT accept ExplanationGenerator.
    pipeline = (
        ExplanationPipeline(
            preprocessor=preprocessor,
            detector=detector,
            engine=counterfactual_engine,
            feasibility_checker=feasibility_checker,
            plausibility_checker=plausibility_checker,
        )
    )

    print(
        "Pipeline initialized successfully."
    )

    # ========================================================================
    # DETAILED COUNTERFACTUAL ANALYSIS
    # ========================================================================

    print_section(
        "DETAILED COUNTERFACTUAL ANALYSIS"
    )

    print(
        f"Running detailed analysis on "
        f"{len(selected_rows)} representative anomalies."
    )

    skipped_count = (
        anomaly_count
        - len(selected_rows)
    )

    print(
        f"Skipping expensive CF search for "
        f"{skipped_count} other detected anomalies."
    )

    detailed_results = []

    detailed_start = (
        time.perf_counter()
    )

    for case_number, row_index in enumerate(
        selected_rows,
        start=1,
    ):

        print()
        print("-" * 78)

        print(
            f"CASE "
            f"{case_number}/"
            f"{len(selected_rows)}"
        )

        print(
            f"Dataset row : "
            f"{row_index}"
        )

        print(
            f"IF score    : "
            f"{anomaly_scores[row_index]:.6f}"
        )

        print("-" * 78)

        # Exact detector input state.
        original_state = (
            test_df.loc[
                [row_index],
                DETECTOR_FEATURES,
            ].copy()
        )

        case_start = (
            time.perf_counter()
        )

        try:

            # Actual ExplanationPipeline API:
            # analyze(sample)
            result = pipeline.analyze(
                original_state
            )

            case_time = (
                time.perf_counter()
                - case_start
            )

            detailed_results.append(
                {
                    "row_index": row_index,
                    "result": result,
                    "case_time": case_time,
                }
            )

            # ---------------------------------------------------------------
            # Extract common result fields
            # ---------------------------------------------------------------

            found = result.get(
                "found"
            )

            reason = result.get(
                "reason"
            )

            original_score = (
                get_result_value(
                    result,
                    "original_score",
                )
            )

            counterfactual_score = (
                get_result_value(
                    result,
                    "counterfactual_score",
                    "cf_score",
                )
            )

            original_label = (
                get_result_value(
                    result,
                    "original_label",
                )
            )

            counterfactual_label = (
                get_result_value(
                    result,
                    "counterfactual_label",
                    "cf_label",
                )
            )

            sparsity = result.get(
                "sparsity"
            )

            normalized_proximity = (
                result.get(
                    "normalized_proximity"
                )
            )

            feasibility_status = (
                result.get(
                    "feasibility_status"
                )
            )

            plausibility_status = (
                result.get(
                    "plausibility_status"
                )
            )

            plausibility_distance = (
                result.get(
                    "plausibility_distance"
                )
            )

            changed_features = (
                result.get(
                    "changed_features"
                )
            )

            # ---------------------------------------------------------------
            # Print result
            # ---------------------------------------------------------------

            print(
                f"Found               : "
                f"{found}"
            )

            print(
                f"Reason              : "
                f"{reason}"
            )

            print(
                f"Original label      : "
                f"{original_label}"
            )

            print(
                f"Counterfactual label: "
                f"{counterfactual_label}"
            )

            print(
                f"Original score      : "
                f"{original_score}"
            )

            print(
                f"CF score            : "
                f"{counterfactual_score}"
            )

            print(
                f"Sparsity            : "
                f"{sparsity}"
            )

            print(
                f"Proximity           : "
                f"{normalized_proximity}"
            )

            print(
                f"Feasibility         : "
                f"{feasibility_status}"
            )

            print(
                f"Plausibility        : "
                f"{plausibility_status}"
            )

            if plausibility_distance is not None:

                print(
                    f"Plausibility distance: "
                    f"{plausibility_distance}"
                )

            if changed_features is not None:

                print(
                    f"Changed features    : "
                    f"{changed_features}"
                )

            # ---------------------------------------------------------------
            # Print counterfactual feature changes
            # ---------------------------------------------------------------

            counterfactual_state = (
                get_result_value(
                    result,
                    "counterfactual_state",
                    "counterfactual",
                    "best_counterfactual",
                )
            )

            print_feature_changes(
                original_state,
                counterfactual_state,
            )

            # ---------------------------------------------------------------
            # Runtime
            # ---------------------------------------------------------------

            print()

            print(
                f"Case runtime: "
                f"{case_time:.3f} sec"
            )

        except Exception as exc:

            case_time = (
                time.perf_counter()
                - case_start
            )

            print()
            print(
                "PIPELINE ERROR"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            detailed_results.append(
                {
                    "row_index": row_index,
                    "result": None,
                    "case_time": case_time,
                    "error": str(exc),
                }
            )

    detailed_time = (
        time.perf_counter()
        - detailed_start
    )

    # ========================================================================
    # DETAILED CASE SUMMARY
    # ========================================================================

    print_section(
        "DETAILED CASE SUMMARY"
    )

    analyzed_count = len(
        detailed_results
    )

    found_count = 0
    feasible_count = 0
    plausible_count = 0
    error_count = 0

    for item in detailed_results:

        result = item.get(
            "result"
        )

        if result is None:
            error_count += 1
            continue

        if result.get(
            "found"
        ) is True:

            found_count += 1

        if result.get(
            "feasibility_status"
        ) == "feasible":

            feasible_count += 1

        if result.get(
            "plausibility_status"
        ) == "plausible":

            plausible_count += 1

    cf_success_rate = (
        found_count
        / analyzed_count
        * 100
        if analyzed_count > 0
        else 0.0
    )

    feasibility_rate = (
        feasible_count
        / found_count
        * 100
        if found_count > 0
        else 0.0
    )

    plausibility_rate = (
        plausible_count
        / found_count
        * 100
        if found_count > 0
        else 0.0
    )

    print(
        f"Detailed cases analyzed : "
        f"{analyzed_count}"
    )

    print(
        f"Counterfactuals found   : "
        f"{found_count}"
    )

    print(
        f"CF success rate         : "
        f"{cf_success_rate:.2f}%"
    )

    print(
        f"Feasible CFs            : "
        f"{feasible_count}"
    )

    if found_count > 0:

        print(
            f"Feasibility rate        : "
            f"{feasibility_rate:.2f}%"
        )

    print(
        f"Plausible CFs           : "
        f"{plausible_count}"
    )

    if found_count > 0:

        print(
            f"Plausibility rate       : "
            f"{plausibility_rate:.2f}%"
        )

    print(
        f"Pipeline errors         : "
        f"{error_count}"
    )

    print(
        f"Detailed CF runtime     : "
        f"{detailed_time:.3f} sec"
    )

    if analyzed_count > 0:

        print(
            f"Average case runtime    : "
            f"{detailed_time / analyzed_count:.3f} sec"
        )

    # ========================================================================
    # CHANGED FEATURE SUMMARY
    # ========================================================================

    changed_feature_counts = {}

    for item in detailed_results:

        result = item.get(
            "result"
        )

        if result is None:
            continue

        if result.get(
            "found"
        ) is not True:
            continue

        changed_features = result.get(
            "changed_features"
        )

        if not changed_features:
            continue

        if isinstance(
            changed_features,
            str,
        ):

            changed_features = [
                changed_features
            ]

        for feature_item in (
            changed_features
        ):

            if isinstance(
                feature_item,
                dict,
            ):

                feature_name = (
                    feature_item.get(
                        "feature",
                        "Unknown",
                    )
                )

            else:

                feature_name = str(
                    feature_item
                )

            changed_feature_counts[
                feature_name
            ] = (
                changed_feature_counts.get(
                    feature_name,
                    0,
                )
                + 1
            )

    if changed_feature_counts:

        print()
        print(
            "Changed-feature frequency:"
        )

        for feature, count in sorted(
            changed_feature_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):

            percentage = (
                count
                / found_count
                * 100
                if found_count > 0
                else 0.0
            )

            print(
                f"  {feature}: "
                f"{count}/{found_count} "
                f"({percentage:.2f}%)"
            )

    # ========================================================================
    # FINAL DATASET SUMMARY
    # ========================================================================

    print_section(
        "FINAL DATASET SUMMARY"
    )

    print(
        f"Dataset                  : "
        f"{input_path.name}"
    )

    print(
        f"Total records            : "
        f"{total_records}"
    )

    print(
        f"Normal records           : "
        f"{normal_count}"
    )

    print(
        f"Anomalous records        : "
        f"{anomaly_count}"
    )

    print(
        f"Anomaly rate             : "
        f"{anomaly_rate:.2f}%"
    )

    print(
        f"Detection time           : "
        f"{total_detection_time:.4f} sec"
    )

    print(
        f"Detailed cases           : "
        f"{analyzed_count}"
    )

    print(
        f"Counterfactuals found    : "
        f"{found_count}"
    )

    print(
        f"Detailed CF success rate : "
        f"{cf_success_rate:.2f}%"
    )

    print(
        f"Feasible CFs             : "
        f"{feasible_count}"
    )

    print(
        f"Plausible CFs            : "
        f"{plausible_count}"
    )

    print(
        f"Detailed CF runtime      : "
        f"{detailed_time:.3f} sec"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The synthetic anomaly rate is a model output on "
        "unseen synthetic data. It is NOT a real-world "
        "machine-failure rate."
    )

    print(
        "Counterfactual changes indicate changes that move "
        "the observation into the detector's learned normal "
        "region. They are not proof of causal or physical "
        "root causes."
    )

    print_section(
        "TEST COMPLETE"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()