import pandas as pd

from src.preprocessing import DataPreprocessor
from src.detector import AnomalyDetector
from src.evaluator import Evaluator


DATA_PATH = "data/raw/ai4i2020.csv"
FEATURES_CONFIG_PATH = "config/features.yaml"
MODEL_CONFIG_PATH = "config/model_config.yaml"


print("=" * 70)
print("AI4I 2020 - FROZEN DETECTOR EVALUATION")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load raw AI4I dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print(f"\nFull dataset shape: {df.shape}")


# ---------------------------------------------------------
# 2. Reproduce the official project split
# ---------------------------------------------------------
preprocessor = DataPreprocessor(
    features_config_path=FEATURES_CONFIG_PATH,
    model_config_path=MODEL_CONFIG_PATH,
)

normal_train, normal_eval, failure_eval = (
    preprocessor.prepare_datasets(df)
)

print(f"Normal training records   : {len(normal_train)}")
print(f"Normal evaluation records : {len(normal_eval)}")
print(f"Failure evaluation records: {len(failure_eval)}")


# ---------------------------------------------------------
# 3. Load the frozen scaler
# ---------------------------------------------------------
preprocessor = DataPreprocessor.load(
    path="models/scaler.joblib",
    features_config_path=FEATURES_CONFIG_PATH,
    model_config_path=MODEL_CONFIG_PATH,
)


# ---------------------------------------------------------
# 4. Load the frozen Isolation Forest
# ---------------------------------------------------------
detector = AnomalyDetector.load(
    path="models/isolation_forest.joblib",
    features_config_path=FEATURES_CONFIG_PATH,
    model_config_path=MODEL_CONFIG_PATH,
)


# ---------------------------------------------------------
# 5. Combine held-out normal + known failure evaluation data
# ---------------------------------------------------------
evaluation_df = pd.concat(
    [normal_eval, failure_eval],
    ignore_index=True,
)

print(f"\nEvaluation records: {len(evaluation_df)}")


# ---------------------------------------------------------
# 6. Prepare detector features
# ---------------------------------------------------------
X_eval = evaluation_df[
    preprocessor.feature_names
]


# ---------------------------------------------------------
# 7. Get true Machine failure labels
# ---------------------------------------------------------
y_true = (
    evaluation_df["Machine failure"]
    .astype(int)
    .to_numpy()
)


# ---------------------------------------------------------
# 8. Transform using the frozen scaler
# ---------------------------------------------------------
X_scaled = preprocessor.transform(X_eval)


# ---------------------------------------------------------
# 9. Run frozen detector
# ---------------------------------------------------------
y_pred = detector.predict_labels(X_scaled)


# ---------------------------------------------------------
# 10. Evaluate detector
# ---------------------------------------------------------
evaluator = Evaluator()

metrics = evaluator.evaluate_detector(
    y_true=y_true,
    y_pred=y_pred,
)


# ---------------------------------------------------------
# 11. Print results
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("DETECTOR RESULTS")
print("=" * 70)

print(f"Precision : {metrics['precision']:.4f}")
print(f"Recall    : {metrics['recall']:.4f}")
print(f"F1 Score  : {metrics['f1_score']:.4f}")

print("\nConfusion Matrix")
print("                 Predicted")
print("                 Normal   Anomaly")
print(
    f"Actual Normal    {metrics['confusion_matrix'][0][0]:>6}"
    f"   {metrics['confusion_matrix'][0][1]:>7}"
)
print(
    f"Actual Failure   {metrics['confusion_matrix'][1][0]:>6}"
    f"   {metrics['confusion_matrix'][1][1]:>7}"
)

print("\n" + "=" * 70)
print("Evaluation complete.")
print("=" * 70)