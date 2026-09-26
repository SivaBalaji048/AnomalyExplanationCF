"""
Real AI4I counterfactual evaluation.

Evaluates the complete explanation pipeline on real AI4I 2020
records that the frozen Isolation Forest classifies as anomalies.

The evaluation reports:
- counterfactuals found
- validity
- feasibility
- plausibility
- sparsity
- normalized proximity
- generation time
- changed features and deltas
"""

from pathlib import Path

import pandas as pd

from src.preprocessing import DataPreprocessor
from src.detector import AnomalyDetector
from src.counterfactual import CounterfactualEngine
from src.feasibility import FeasibilityChecker
from src.plausibility import PlausibilityChecker
from src.pipeline import ExplanationPipeline
from src.evaluator import Evaluator


# ============================================================
# Paths
# ============================================================

DATA_PATH = Path("data/raw/ai4i2020.csv")
FEATURES_CONFIG_PATH = Path("config/features.yaml")
MODEL_CONFIG_PATH = Path("config/model_config.yaml")

SCALER_PATH = Path("models/scaler.joblib")
MODEL_PATH = Path("models/isolation_forest.joblib")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 75)
    print("REAL AI4I 2020 - COUNTERFACTUAL EVALUATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Load raw AI4I dataset
    # --------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    print(f"\nFull dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 2. Reproduce the official project split
    # --------------------------------------------------------
    split_preprocessor = DataPreprocessor(
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )

    normal_train, normal_eval, failure_eval = (
        split_preprocessor.prepare_datasets(df)
    )

    print(f"Normal training records    : {len(normal_train)}")
    print(f"Normal evaluation records  : {len(normal_eval)}")
    print(f"Failure evaluation records : {len(failure_eval)}")

    # --------------------------------------------------------
    # 3. Load the frozen fitted scaler
    # --------------------------------------------------------
    preprocessor = DataPreprocessor.load(
        path=SCALER_PATH,
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )

    # --------------------------------------------------------
    # 4. Load the frozen Isolation Forest
    # --------------------------------------------------------
    detector = AnomalyDetector.load(
        path=MODEL_PATH,
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )

    # --------------------------------------------------------
    # 5. Identify evaluation populations
    #
    # Population A: All detector-predicted anomalies in the evaluation population
    # Population B: Primary counterfactual evaluation cases:
    #               known failures that the detector predicted as anomalies
    # --------------------------------------------------------
    evaluation_df = pd.concat(
        [normal_eval, failure_eval],
        ignore_index=True,
    )

    X_eval = evaluation_df[
        preprocessor.feature_names
    ]

    X_scaled = preprocessor.transform(X_eval)

    predicted_labels = detector.predict_labels(X_scaled)

    anomaly_mask = predicted_labels == -1

    # Population A: All detector-predicted anomalies in the evaluation population
    all_anomalies_df = evaluation_df.loc[
        anomaly_mask.to_numpy()
    ].copy()

    # Composition of Population A
    known_failures_count = int(
        (all_anomalies_df["Machine failure"] == 1).sum()
    )
    known_normal_count = int(
        (all_anomalies_df["Machine failure"] == 0).sum()
    )

    # Population B: Primary counterfactual evaluation cases
    # Known failures that the detector predicted as anomalies
    primary_cf_cases_df = all_anomalies_df[
        all_anomalies_df["Machine failure"] == 1
    ].copy()

    print("\n" + "=" * 75)
    print("EVALUATION POPULATIONS")
    print("=" * 75)
    print("Population A: All detector-predicted anomalies in the evaluation population")
    print("Population B: Primary counterfactual evaluation cases:")
    print("              known failures that the detector predicted as anomalies")
    print("-" * 75)
    print(f"Total evaluation records                : {len(evaluation_df)}")
    print(f"All detector-predicted anomalies        : {len(all_anomalies_df)}")
    print(f"  Known failures among them             : {known_failures_count}")
    print(f"  Known normal records among them       : {known_normal_count}")
    print(f"Primary counterfactual evaluation cases : {len(primary_cf_cases_df)}")

    if primary_cf_cases_df.empty:
        print("\nNo primary counterfactual evaluation cases were found.")
        print("Counterfactual evaluation cannot continue.")
        return

    # --------------------------------------------------------
    # 7. Build plausibility reference
    #
    # IMPORTANT:
    # Use the exact normal-training partition that the
    # project uses for detector training.
    # --------------------------------------------------------
    normal_training_features = normal_train[
        preprocessor.feature_names
    ].copy()

    plausibility_checker = PlausibilityChecker(
        normal_training_data=normal_training_features,
        preprocessor=preprocessor,
        config_path=str(FEATURES_CONFIG_PATH),
    )

    # --------------------------------------------------------
    # 8. Build counterfactual components
    # --------------------------------------------------------
    engine = CounterfactualEngine(
        features_config_path=str(FEATURES_CONFIG_PATH),
        model_config_path=str(MODEL_CONFIG_PATH),
    )

    feasibility_checker = FeasibilityChecker(
        config_path=str(FEATURES_CONFIG_PATH),
    )

    pipeline = ExplanationPipeline(
        preprocessor=preprocessor,
        detector=detector,
        engine=engine,
        feasibility_checker=feasibility_checker,
        plausibility_checker=plausibility_checker,
    )

    evaluator = Evaluator()

    # --------------------------------------------------------
    # 9. Run counterfactual analysis
    # --------------------------------------------------------
    pipeline_results = []

    print("\n" + "=" * 75)
    print("RUNNING COUNTERFACTUAL ANALYSIS")
    print("=" * 75)

    for case_number, (index, row) in enumerate(
        primary_cf_cases_df.iterrows(),
        start=1,
    ):
        sample = pd.DataFrame(
            [row[preprocessor.feature_names].to_dict()],
            columns=preprocessor.feature_names,
        ).astype(float)

        # Preserve original AI4I row identifier for reporting.
        original_row_index = index

        print(
            f"\n[{case_number}/{len(primary_cf_cases_df)}] "
            f"AI4I evaluation row index: {original_row_index}"
        )

        try:
            result = pipeline.analyze(sample)

            # Keep source information for later reporting.
            result["source_row_index"] = int(original_row_index)
            result["machine_failure_label"] = int(
                row["Machine failure"]
            )

            pipeline_results.append(result)

            # ------------------------------------------------
            # Print case result
            # ------------------------------------------------
            print(
                f"  Evaluation row index  : {original_row_index}"
            )
            print(
                f"  Counterfactual found  : {result.get('found')}"
            )
            print(
                f"  Reason                : {result.get('reason')}"
            )
            print(
                f"  Candidates evaluated  : {result.get('candidates_evaluated')}"
            )
            print(
                f"  Generation time       : {result.get('generation_time_seconds')}"
            )

            if result.get("found"):
                print(
                    f"  Changed features      : {result.get('changed_features')}"
                )
                print(
                    f"  Sparsity              : {result.get('sparsity')}"
                )
                print(
                    f"  Normalized proximity  : {result.get('normalized_proximity')}"
                )
                print(
                    f"  Feasibility status    : {result.get('feasibility_status')}"
                )
                print(
                    f"  Plausibility status   : {result.get('plausibility_status')}"
                )
                print(
                    f"  Plausibility distance : {result.get('plausibility_distance')}"
                )
                print(
                    f"  Plausibility threshold: {result.get('plausibility_threshold')}"
                )

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            print(
                f"  Evaluation row index  : {original_row_index}"
            )
            print(
                f"  Counterfactual found  : False"
            )
            print(
                f"  Reason                : pipeline_error"
            )
            print(
                f"  Candidates evaluated  : 0"
            )
            print(
                f"  Generation time       : 0.0"
            )

            # Keep evaluation running for the remaining cases.
            pipeline_results.append(
                {
                    "source_row_index": int(original_row_index),
                    "machine_failure_label": int(
                        row["Machine failure"]
                    ),
                    "found": False,
                    "reason": "pipeline_error",
                    "original_label": -1,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "candidates_evaluated": 0,
                    "generation_time_seconds": 0.0,
                }
            )

    # --------------------------------------------------------
    # 10. Aggregate evaluation
    # --------------------------------------------------------
    metrics = evaluator.evaluate_counterfactual_batch(
        pipeline_results
    )

    # --------------------------------------------------------
    # 11. Print aggregate results
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("COUNTERFACTUAL EVALUATION RESULTS")
    print("=" * 75)

    print(
        f"Total cases analyzed             : "
        f"{metrics['total_cases']}"
    )

    print(
        f"Anomalies analyzed               : "
        f"{metrics['anomalies_analyzed']}"
    )

    print(
        f"Counterfactuals found            : "
        f"{metrics['counterfactuals_found']}"
    )

    print(
        f"Valid counterfactuals            : "
        f"{metrics['valid_counterfactuals']}"
    )

    print(
        f"Counterfactual validity rate     : "
        f"{metrics['counterfactual_validity_rate']:.4f}"
    )

    print(
        f"Feasible counterfactuals         : "
        f"{metrics['feasible_counterfactuals']}"
    )

    print(
        f"Feasibility rate                 : "
        f"{metrics['feasibility_rate']:.4f}"
    )

    print(
        f"Plausible counterfactuals        : "
        f"{metrics['plausible_counterfactuals']}"
    )

    print(
        f"Plausibility rate                : "
        f"{metrics['plausibility_rate']:.4f}"
    )

    print(
        f"Average sparsity                 : "
        f"{metrics['average_sparsity']}"
    )

    print(
        f"Average normalized proximity     : "
        f"{metrics['average_normalized_proximity']}"
    )

    print(
        f"Average generation time (sec)    : "
        f"{metrics['average_generation_time_seconds']}"
    )

    # --------------------------------------------------------
    # 12. Additional breakdown
    # --------------------------------------------------------
    successful_results = [
        result
        for result in pipeline_results
        if result.get("found")
    ]

    print("\n" + "=" * 75)
    print("ADDITIONAL BREAKDOWN")
    print("=" * 75)

    print(
        f"Found counterfactuals : "
        f"{len(successful_results)} / {len(pipeline_results)}"
    )

    if successful_results:

        feasible_count = sum(
            result.get("feasibility_status") == "feasible"
            for result in successful_results
        )

        plausible_count = sum(
            result.get("plausibility_status") == "plausible"
            for result in successful_results
        )

        print(
            f"Found CFs feasible   : "
            f"{feasible_count} / {len(successful_results)}"
        )

        print(
            f"Found CFs plausible  : "
            f"{plausible_count} / {len(successful_results)}"
        )

        changed_feature_counts = {}

        for result in successful_results:
            for feature_detail in result.get(
                "changed_features",
                [],
            ):
                feature = feature_detail["feature"]
                changed_feature_counts[feature] = (
                    changed_feature_counts.get(feature, 0) + 1
                )

        print("\nChanged feature frequency:")

        for feature, count in sorted(
            changed_feature_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(
                f"  {feature}: {count}"
            )

    # --------------------------------------------------------
    # 13. Failure summary (cases without counterfactual)
    # --------------------------------------------------------
    unsuccessful_results = [
        result
        for result in pipeline_results
        if not result.get("found")
    ]

    print("\n" + "=" * 75)
    print("FAILURE SUMMARY (CASES WITHOUT COUNTERFACTUAL)")
    print("=" * 75)

    print(
        f"Total cases without counterfactual: "
        f"{len(unsuccessful_results)} / {len(pipeline_results)}"
    )

    if unsuccessful_results:
        # Failure breakdown by reason
        reason_counts = {}
        for result in unsuccessful_results:
            reason = result.get("reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        print("\nFailure breakdown by reason:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")

        print("\nCase-by-case failure details:")
        for i, result in enumerate(unsuccessful_results, start=1):
            print(f"\n  Case {i}:")
            print(
                f"    Evaluation row index : "
                f"{result.get('source_row_index')}"
            )
            print(
                f"    Reason               : "
                f"{result.get('reason')}"
            )
            print(
                f"    Candidates evaluated : "
                f"{result.get('candidates_evaluated')}"
            )
            print(
                f"    Generation time      : "
                f"{result.get('generation_time_seconds')}"
            )
    else:
        print("All analyzed cases found a valid counterfactual.")

    # --------------------------------------------------------
    # 14. Final status
    # --------------------------------------------------------
    error_count = sum(
        result.get("reason") == "pipeline_error"
        for result in pipeline_results
    )

    print("\n" + "=" * 75)
    print("FINAL STATUS")
    print("=" * 75)

    print(f"Pipeline errors: {error_count}")

    if error_count == 0:
        print("All real AI4I counterfactual cases completed successfully.")
    else:
        print(
            "Some cases encountered pipeline errors. "
            "Review the case-level output above."
        )

    print("=" * 75)


if __name__ == "__main__":
    main()