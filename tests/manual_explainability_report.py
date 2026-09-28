"""
Manual Explainability Report

Terminal-only human-readable explainability report for one 
selected record in the configured evaluation population.
"""

import sys
import argparse
import yaml
from pathlib import Path

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.preprocessing import DataPreprocessor
from src.detector import AnomalyDetector
from src.counterfactual import CounterfactualEngine
from src.feasibility import FeasibilityChecker
from src.plausibility import PlausibilityChecker
from src.pipeline import ExplanationPipeline

DATA_PATH = REPO_ROOT / "data/raw/ai4i2020.csv"
FEATURES_CONFIG_PATH = REPO_ROOT / "config/features.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "config/model_config.yaml"
SCALER_PATH = REPO_ROOT / "models/scaler.joblib"
MODEL_PATH = REPO_ROOT / "models/isolation_forest.joblib"

def main():
    parser = argparse.ArgumentParser(description="Generate explainability report for a specific evaluation row.")
    parser.add_argument(
        "--evaluation-row-index",
        type=int,
        default=1933,
        help="The row index in the evaluation dataset to explain."
    )
    args = parser.parse_args()
    row_idx = args.evaluation_row_index
    
    # 1. Load configurations
    with open(FEATURES_CONFIG_PATH, "r") as f:
        features_config = yaml.safe_load(f)
        
    mutable_features = features_config["counterfactual_policy"]["mutable_features"]
    
    # 2. Load dataset and reconstruct population
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw dataset not found at {DATA_PATH}")
        sys.exit(1)
        
    preprocessor = DataPreprocessor(
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )
    normal_train, normal_eval, failure_eval = preprocessor.prepare_datasets(df)
    evaluation_df = pd.concat([normal_eval, failure_eval], ignore_index=True)
    
    # Validation
    if not (0 <= row_idx < len(evaluation_df)):
        print(f"Error: Row index {row_idx} is out of bounds (0 - {len(evaluation_df) - 1}).")
        sys.exit(1)
        
    row = evaluation_df.iloc[row_idx]
    machine_failure_label = int(row["Machine failure"])
    
    # 3. Load artifacts
    scaler = DataPreprocessor.load(
        path=SCALER_PATH,
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )
    detector = AnomalyDetector.load(
        path=MODEL_PATH,
        features_config_path=FEATURES_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
    )
    
    if not scaler.is_fitted:
        print("Error: Loaded scaler is not fitted.")
        sys.exit(1)
    if not detector.is_fitted:
        print("Error: Loaded detector is not fitted.")
        sys.exit(1)
        
    expected_features = scaler.feature_names
    input_sample = pd.DataFrame([row[expected_features].to_dict()], columns=expected_features).astype(float)
    
    # Check that input feature order matches config
    if list(input_sample.columns) != expected_features:
        print("Error: Feature order mismatch.")
        sys.exit(1)
    
    # 4. Predict
    transformed_sample = scaler.transform(input_sample)
    pred_label = detector.predict_labels(transformed_sample).iloc[0]
    pred_score = detector.predict_anomaly_score(transformed_sample).iloc[0]
    
    is_anomaly = (pred_label == -1)
    classification_str = "Anomalous" if is_anomaly else "Normal"
    
    # 5. Print Output
    print("=" * 80)
    print("MACHINE STATE EXPLAINABILITY REPORT")
    print("=" * 80)
    print("Report context")
    print(f"- Evaluation row index: {row_idx}")
    print(f"- Dataset evaluation label (evaluation only): {machine_failure_label}")
    print("- Frozen artifact status: Loaded and fitted")
    
    print("\n1. Detector Decision")
    print(f"- Detector classification: {classification_str}")
    print(f"- Isolation Forest decision score: {pred_score:.4f}")
    print("- Short score interpretation: Higher values are more normal; lower values are more anomalous.")
    
    print("\n2. Submitted Machine State")
    for feat in expected_features:
        print(f"  - {feat}: {input_sample.iloc[0][feat]}")
        
    print("\n3. Counterfactual Explanation")
    
    if not is_anomaly:
        print("The detector classified this machine state as Normal.")
        print("No counterfactual explanation is required or generated.")
    else:
        # Build explanation pipeline
        normal_training_features = normal_train[expected_features].copy()
        plausibility_checker = PlausibilityChecker(
            normal_training_data=normal_training_features,
            preprocessor=scaler,
            config_path=FEATURES_CONFIG_PATH,
        )
        engine = CounterfactualEngine(
            features_config_path=FEATURES_CONFIG_PATH,
            model_config_path=MODEL_CONFIG_PATH,
        )
        feasibility_checker = FeasibilityChecker(
            config_path=FEATURES_CONFIG_PATH,
        )
        pipeline = ExplanationPipeline(
            preprocessor=scaler,
            detector=detector,
            engine=engine,
            feasibility_checker=feasibility_checker,
            plausibility_checker=plausibility_checker,
        )
        
        result = pipeline.analyze(input_sample)
        
        if result["found"]:
            print(f"{'Feature':<25} | {'Original':<10} | {'Suggested':<10} | {'Change':<10} | Counterfactual policy")
            print("-" * 90)
            
            orig = input_sample.iloc[0]
            cf = result["counterfactual_state"].iloc[0] if isinstance(result["counterfactual_state"], pd.DataFrame) else result["counterfactual_state"]
            
            for feat in expected_features:
                val_orig = float(orig[feat])
                val_cf = float(cf[feat])
                diff = val_cf - val_orig
                
                if feat in mutable_features:
                    policy_str = "Changed" if diff != 0 else "Mutable (unchanged)"
                else:
                    policy_str = "Locked / unchanged"
                    
                diff_str = f"{diff:+.2f}" if diff != 0 else "0.00"
                
                print(f"{feat:<25} | {val_orig:<10.2f} | {val_cf:<10.2f} | {diff_str:<10} | {policy_str}")
                
            print(f"\n- Sparsity: {result.get('sparsity', '')}")
            print(f"- Normalized proximity: {result.get('normalized_proximity', 0.0):.4f}")
            print(f"- Candidates evaluated: {result.get('candidates_evaluated', '')}")
            print(f"- Generation time: {result.get('generation_time_seconds', 0.0):.4f} s")
            
            print("\n4. Trust Checks")
            print("- Detector-valid counterfactual: Yes")
            print(f"- Feasibility status: {result.get('feasibility_status', '')}")
            print(f"- Plausibility status: {result.get('plausibility_status', '')}")
            print(f"- Mahalanobis distance: {result.get('plausibility_distance', 0.0):.4f}")
            print(f"- Plausibility threshold: {result.get('plausibility_threshold', 0.0):.4f}")
            print(f"- Plausibility method: {result.get('plausibility_method', '')}")
            
            if result.get('plausibility_status') != "plausible":
                print("\n  [!] CAUTION: The suggested detector-valid state lies outside the normal")
                print("      training-data plausibility threshold. It may represent a mathematically")
                print("      valid but operationally unrealistic configuration.")
                
        else:
            print("- No counterfactual generated.")
            print(f"- Reason: {result.get('reason', 'Unknown')}")
            print(f"- Candidates evaluated: {result.get('candidates_evaluated', 0)}")
            print(f"- Generation time: {result.get('generation_time_seconds', 0.0):.4f} s")
            print("- No suggested operating change.")
            
    print("\n" + "=" * 80)
    print("Responsible-Use Notice")
    print("- This is a model-based explanation, not a causal root-cause conclusion.")
    print("- It is not a certified-safe operating instruction.")
    print("- It does not guarantee machine repair or failure prevention.")
    print("- Empirical feature bounds are not physical safety limits.")
    print("=" * 80)

if __name__ == "__main__":
    main()
