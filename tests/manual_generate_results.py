"""
Generate final reports and plots in the results/ directory
using frozen project artifacts.

This script executes the counterfactual evaluation using the 
frozen baseline and saves the resulting metrics, data, and plots.
"""

import sys
import json
from pathlib import Path

# --------------------------------------------------------
# Resolve Repository Root and Paths
# --------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.preprocessing import DataPreprocessor
from src.detector import AnomalyDetector
from src.counterfactual import CounterfactualEngine
from src.feasibility import FeasibilityChecker
from src.plausibility import PlausibilityChecker
from src.pipeline import ExplanationPipeline
from src.evaluator import Evaluator

# Input Paths
DATA_PATH = REPO_ROOT / "data/raw/ai4i2020.csv"
FEATURES_CONFIG_PATH = REPO_ROOT / "config/features.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "config/model_config.yaml"
SCALER_PATH = REPO_ROOT / "models/scaler.joblib"
MODEL_PATH = REPO_ROOT / "models/isolation_forest.joblib"

# Output Paths
RESULTS_METRICS_DIR = REPO_ROOT / "results/metrics"
RESULTS_EXPLANATIONS_DIR = REPO_ROOT / "results/explanations"
RESULTS_PLOTS_DIR = REPO_ROOT / "results/plots"


def main():
    print("=" * 75)
    print("GENERATING RESULTS: AUTOMATED ANOMALY EXPLANATION WITH COUNTERFACTUALS")
    print("=" * 75)

    # 1. Create output directories
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # 3. Prep splits (to get evaluation records and normal training data)
    split_preprocessor = DataPreprocessor(
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )
    normal_train, normal_eval, failure_eval = split_preprocessor.prepare_datasets(df)
    
    # 4. Load frozen components
    preprocessor = DataPreprocessor.load(
        path=SCALER_PATH,
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )
    detector = AnomalyDetector.load(
        path=MODEL_PATH,
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )
    
    # 5. Build evaluation DF
    evaluation_df = pd.concat([normal_eval, failure_eval], ignore_index=True)
    X_eval = evaluation_df[preprocessor.feature_names]
    X_scaled = preprocessor.transform(X_eval)
    
    # 6. Detector Predictions
    predicted_labels = detector.predict_labels(X_scaled)
    anomaly_scores = detector.predict_anomaly_score(X_scaled)
    anomaly_mask = predicted_labels == -1
    
    evaluation_df["detector_prediction"] = predicted_labels
    evaluation_df["detector_anomaly_score"] = anomaly_scores
    
    # True and Pred labels for metrics (1 = Anomaly/Machine Failure, 0 = Normal)
    y_true = evaluation_df["Machine failure"].values
    y_pred = anomaly_mask.astype(int)
    
    # Calculate Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Extract Populations
    all_anomalies_df = evaluation_df.loc[anomaly_mask].copy()
    known_failures_count = int((all_anomalies_df["Machine failure"] == 1).sum())
    known_normal_count = int((all_anomalies_df["Machine failure"] == 0).sum())
    
    primary_cf_cases_df = all_anomalies_df[all_anomalies_df["Machine failure"] == 1].copy()
    
    # 7. Prepare Detector Metrics
    detector_metrics = {
        "evaluation_population_description": "held-out normal records plus all known machine-failure records",
        "evaluation_record_count": len(evaluation_df),
        "all_detector_predicted_anomalies": len(all_anomalies_df),
        "known_failures_among_anomalies": known_failures_count,
        "known_normal_records_among_anomalies": known_normal_count,
        "primary_counterfactual_evaluation_cases": len(primary_cf_cases_df),
        "performance": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1])
            }
        }
    }
    
    if primary_cf_cases_df.empty:
        print("No primary counterfactual evaluation cases found. Halting.")
        return
        
    # 8. Setup Counterfactual Pipeline
    normal_training_features = normal_train[preprocessor.feature_names].copy()
    plausibility_checker = PlausibilityChecker(
        normal_training_data=normal_training_features,
        preprocessor=preprocessor,
        config_path=str(FEATURES_CONFIG_PATH),
    )
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
    
    # 9. Execute Pipeline on Primary CF Cases
    pipeline_results = []
    
    for index, row in primary_cf_cases_df.iterrows():
        sample = pd.DataFrame(
            [row[preprocessor.feature_names].to_dict()],
            columns=preprocessor.feature_names,
        ).astype(float)
        
        try:
            result = pipeline.analyze(sample)
            result["evaluation_row_index"] = int(index)
            result["true_machine_failure"] = int(row["Machine failure"])
            result["detector_prediction"] = int(row["detector_prediction"])
            result["detector_anomaly_score"] = float(row["detector_anomaly_score"])
            
            # Extract features for CSV reporting
            for fn in preprocessor.feature_names:
                result[f"original_{fn}"] = float(row[fn])
                if result.get("found"):
                    result[f"counterfactual_{fn}"] = float(result["counterfactual_state"].iloc[0][fn])
                else:
                    result[f"counterfactual_{fn}"] = None
                    
            pipeline_results.append(result)
        except Exception as exc:
            result = {
                "evaluation_row_index": int(index),
                "true_machine_failure": int(row["Machine failure"]),
                "detector_prediction": int(row["detector_prediction"]),
                "detector_anomaly_score": float(row["detector_anomaly_score"]),
                "original_label": -1,
                "found": False,
                "reason": "pipeline_error",
                "error_message": str(exc),
                "candidates_evaluated": 0,
                "generation_time_seconds": 0.0
            }
            for fn in preprocessor.feature_names:
                result[f"original_{fn}"] = float(row[fn])
                result[f"counterfactual_{fn}"] = None
            pipeline_results.append(result)
            
    # 10. Aggregate Metrics
    metrics = evaluator.evaluate_counterfactual_batch(pipeline_results)
    successful_results = [r for r in pipeline_results if r.get("found")]
    cf_not_found = sum(1 for r in pipeline_results if not r.get("found"))
    
    exact_no_cf_indices = [r["evaluation_row_index"] for r in pipeline_results if not r.get("found")]
    
    changed_feature_counts = {}
    for result in successful_results:
        for feature_detail in result.get("changed_features", []):
            feature = feature_detail["feature"]
            changed_feature_counts[feature] = changed_feature_counts.get(feature, 0) + 1
            
    cf_metrics = {
        "primary_counterfactual_evaluation_cases": len(primary_cf_cases_df),
        "counterfactuals_found": metrics.get("counterfactuals_found", 0),
        "counterfactuals_not_found": cf_not_found,
        "found_rate_description": "conditional on known failures detected as anomalous",
        "counterfactual_found_rate_conditional_on_detected_known_failures": float(metrics.get("counterfactuals_found", 0) / len(primary_cf_cases_df)),
        "valid_counterfactuals": metrics.get("valid_counterfactuals", 0),
        "counterfactual_validity_rate_conditional_on_counterfactuals_found": metrics.get("counterfactual_validity_rate", 0.0),
        "feasible_counterfactuals": metrics.get("feasible_counterfactuals", 0),
        "feasibility_rate_conditional_on_counterfactuals_found": metrics.get("feasibility_rate", 0.0),
        "plausible_counterfactuals": metrics.get("plausible_counterfactuals", 0),
        "plausibility_rate_description": "conditional on counterfactuals found",
        "plausibility_rate_conditional_on_counterfactuals_found": metrics.get("plausibility_rate", 0.0),
        "average_sparsity": metrics.get("average_sparsity", 0.0),
        "average_normalized_proximity": metrics.get("average_normalized_proximity", 0.0),
        "average_generation_time_seconds": metrics.get("average_generation_time_seconds", 0.0),
        "changed_feature_counts": changed_feature_counts,
        "plausibility_method": "Mahalanobis distance",
        "plausibility_threshold": next((r.get("plausibility_threshold") for r in pipeline_results if r.get("found")), None),
        "exact_no_counterfactual_evaluation_row_indices": exact_no_cf_indices,
        "limitations": [
            "Counterfactuals are not causal root-cause conclusions.",
            "Suggested changes are not certified-safe actions.",
            "Suggested changes do not guarantee a real machine repair or failure prevention.",
            "Results depend on the frozen detector, current data distribution, and configured constraints."
        ]
    }
    
    # 11. Prepare Case-by-Case CSV
    csv_rows = []
    for r in pipeline_results:
        csv_row = {
            "evaluation_row_index": r.get("evaluation_row_index", ""),
            "true_machine_failure": r.get("true_machine_failure", ""),
            "detector_prediction": r.get("detector_prediction", ""),
            "detector_anomaly_score": r.get("detector_anomaly_score", ""),
            "counterfactual_found": r.get("found", False),
            "failure_reason": r.get("reason", ""),
            "candidates_evaluated": r.get("candidates_evaluated", 0),
            "generation_time_seconds": r.get("generation_time_seconds", 0.0),
            "changed_features": json.dumps(r.get("changed_features", [])) if r.get("found") else "[]",
            "sparsity": r.get("sparsity", ""),
            "normalized_proximity": r.get("normalized_proximity", ""),
            "feasibility_status": r.get("feasibility_status", ""),
            "plausibility_status": r.get("plausibility_status", ""),
            "plausibility_distance": r.get("plausibility_distance", ""),
            "plausibility_threshold": r.get("plausibility_threshold", ""),
        }
        for fn in preprocessor.feature_names:
            csv_row[f"original_{fn}"] = r.get(f"original_{fn}", "")
            cf_val = r.get(f"counterfactual_{fn}")
            csv_row[f"counterfactual_{fn}"] = cf_val if cf_val is not None else ""
        csv_rows.append(csv_row)
        
    # 12. Extract Outcome Categories
    found_and_plausible = sum(1 for r in pipeline_results if r.get("found") and r.get("plausibility_status") == "plausible")
    found_but_implausible = sum(1 for r in pipeline_results if r.get("found") and r.get("plausibility_status") == "implausible")
    not_found = sum(1 for r in pipeline_results if not r.get("found"))
    
    # --------------------------------------------------------
    # VALIDATION CHECKS
    # --------------------------------------------------------
    assert preprocessor.is_fitted, "Preprocessor is not fitted; outputs must be derived from the loaded frozen artifact."
    assert detector.is_fitted, "Detector is not fitted; outputs must be derived from the loaded frozen artifact."
    assert len(csv_rows) == len(primary_cf_cases_df), "Mismatch: CSV row count != primary evaluation case count."
    assert (metrics.get("counterfactuals_found", 0) + cf_not_found) == len(primary_cf_cases_df), "Mismatch: found + not_found != primary cases."
    assert (found_and_plausible + found_but_implausible + not_found) == len(primary_cf_cases_df), "Mismatch: Sum of outcome categories != primary cases."
    
    for r in csv_rows:
        if not r["counterfactual_found"]:
            for fn in preprocessor.feature_names:
                assert r[f"counterfactual_{fn}"] == "", f"Counterfactual feature value incorrectly populated for not-found row {r['evaluation_row_index']}."

    # --------------------------------------------------------
    # WRITE ARTIFACTS
    # --------------------------------------------------------
    # Write Detector Metrics JSON
    with open(RESULTS_METRICS_DIR / "detector_metrics.json", "w") as f:
        json.dump(detector_metrics, f, indent=4)

    # Write CF Metrics JSON
    with open(RESULTS_METRICS_DIR / "counterfactual_metrics.json", "w") as f:
        json.dump(cf_metrics, f, indent=4)
        
    # Write CSV
    pd.DataFrame(csv_rows).to_csv(RESULTS_EXPLANATIONS_DIR / "counterfactual_case_results.csv", index=False)

    # Plot 1: Detector Confusion Matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title("Detector Confusion Matrix (Eval Split)", pad=20)
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal (0)', 'Anomaly (1)'])
    ax.set_yticklabels(['Normal (0)', 'Anomaly (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:d}'.format(z), ha='center', va='center',
                color="white" if z > (cm.max()/2) else "black")
    plt.tight_layout()
    plt.savefig(RESULTS_PLOTS_DIR / "detector_confusion_matrix.png")
    plt.close()

    # Plot 2: Counterfactual Outcome Breakdown
    categories = ["Found and plausible", "Found but implausible", "Not found"]
    counts = [found_and_plausible, found_but_implausible, not_found]
    colors = ["forestgreen", "gold", "crimson"]
        
    fig, ax = plt.subplots()
    bars = ax.bar(categories, counts, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Counterfactual Outcome Breakdown")
    plt.xticks(rotation=45, ha='right')
    ax.bar_label(bars)
    plt.tight_layout()
    plt.savefig(RESULTS_PLOTS_DIR / "counterfactual_outcome_breakdown.png")
    plt.close()
    
    # Plot 3: Changed Feature Frequency
    if changed_feature_counts:
        fig, ax = plt.subplots()
        sorted_features = sorted(changed_feature_counts.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_features]
        f_counts = [x[1] for x in sorted_features]
        
        bars = ax.bar(features, f_counts, color='steelblue')
        ax.set_ylabel("Frequency (Cases)")
        ax.set_title("Changed Feature Frequency in Valid Counterfactuals")
        plt.xticks(rotation=45, ha='right')
        ax.bar_label(bars)
        plt.tight_layout()
        plt.savefig(RESULTS_PLOTS_DIR / "changed_feature_frequency.png")
        plt.close()
    else:
        fig, ax = plt.subplots()
        ax.set_title("Changed Feature Frequency (No Data)")
        plt.tight_layout()
        plt.savefig(RESULTS_PLOTS_DIR / "changed_feature_frequency.png")
        plt.close()
    
    # Plot 4: Plausibility Distance Distribution
    distances = [r.get("plausibility_distance") for r in successful_results if r.get("plausibility_distance") is not None]
    if distances:
        fig, ax = plt.subplots()
        ax.hist(distances, bins=15, edgecolor='black', color='mediumpurple')
        ax.set_xlabel("Mahalanobis Distance")
        ax.set_ylabel("Frequency")
        ax.set_title("Plausibility Distance Distribution of Valid Counterfactuals")
        
        threshold = next((r.get("plausibility_threshold") for r in pipeline_results if r.get("found")), None)
        if threshold is not None:
            ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2, 
                       label=f'Threshold ({threshold:.2f})')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(RESULTS_PLOTS_DIR / "plausibility_distance_distribution.png")
        plt.close()
    else:
        fig, ax = plt.subplots()
        ax.set_title("Plausibility Distance Distribution (No Data)")
        plt.tight_layout()
        plt.savefig(RESULTS_PLOTS_DIR / "plausibility_distance_distribution.png")
        plt.close()
        
    # Plot 5: Generation Time Distribution
    times = [r.get("generation_time_seconds", 0.0) for r in pipeline_results]
    if times:
        fig, ax = plt.subplots()
        ax.hist(times, bins=15, edgecolor='black', color='coral')
        ax.set_xlabel("Generation Time (Seconds)")
        ax.set_ylabel("Frequency")
        ax.set_title("Counterfactual Generation Time Distribution")
        plt.tight_layout()
        plt.savefig(RESULTS_PLOTS_DIR / "generation_time_distribution.png")
        plt.close()

    print("\n" + "=" * 75)
    print("RESULTS GENERATION COMPLETE")
    print("=" * 75)
    
    print("\nExact saved file paths:")
    print(f"  - {RESULTS_METRICS_DIR / 'detector_metrics.json'}")
    print(f"  - {RESULTS_METRICS_DIR / 'counterfactual_metrics.json'}")
    print(f"  - {RESULTS_EXPLANATIONS_DIR / 'counterfactual_case_results.csv'}")
    print(f"  - {RESULTS_PLOTS_DIR / 'detector_confusion_matrix.png'}")
    print(f"  - {RESULTS_PLOTS_DIR / 'counterfactual_outcome_breakdown.png'}")
    print(f"  - {RESULTS_PLOTS_DIR / 'changed_feature_frequency.png'}")
    print(f"  - {RESULTS_PLOTS_DIR / 'plausibility_distance_distribution.png'}")
    print(f"  - {RESULTS_PLOTS_DIR / 'generation_time_distribution.png'}")
    
    print(f"\nDetector population counts:")
    print(f"  Evaluation records: {len(evaluation_df)}")
    print(f"  All detector anomalies: {len(all_anomalies_df)}")
    print(f"  Known failures among anomalies: {known_failures_count}")
    print(f"  Known normal records among anomalies: {known_normal_count}")
    
    print(f"\nCounterfactual outcomes:")
    print(f"  Primary counterfactual evaluation cases: {len(primary_cf_cases_df)}")
    print(f"  Found and plausible: {found_and_plausible}")
    print(f"  Found but implausible: {found_but_implausible}")
    print(f"  Not found: {not_found}")
    
    print(f"\nNo-counterfactual evaluation-row indices:")
    print(f"  {exact_no_cf_indices}")
    
    print(f"\nExplicit confirmation:")
    print("  The script successfully loaded frozen artifacts (scaler, isolation forest).")
    print("  The script strictly derived outputs from the loaded artifacts and did not retrain or overwrite them.")
    print("=" * 75)

if __name__ == "__main__":
    main()
