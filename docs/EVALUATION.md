# Evaluation Report

## 1. Evaluation Objective

This evaluation report assesses the performance, validity, feasibility, and distributional plausibility of the Automated Anomaly Explanation system on the AI4I 2020 synthetic predictive-maintenance benchmark dataset. The evaluation addresses two distinct stages:

1. **Anomaly Detector Performance:** Evaluating the frozen Isolation Forest baseline on the held-out evaluation dataset, assessing its ability to flag anomalous operational states against known machine failure labels.
2. **Counterfactual Explanation Pipeline:** Evaluating the counterfactual search, feasibility, and plausibility modules exclusively on the subset of known machine failures that the frozen detector correctly identified as anomalies.

---

## 2. Reproducibility and Frozen Baseline

The evaluation adheres to a strict reproducibility and governance protocol to prevent data leakage and guarantee valid evaluation metrics:

- **Raw Dataset:** AI4I 2020 Predictive Maintenance Dataset (`data/raw/ai4i2020.csv`), comprising 10,000 synthetic-realistic operational records.
- **Approved Detector Inputs:** Strictly five continuous operational features:
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`
- **Data Splitting & Fitting Policy:**
  - Scaler (`models/scaler.joblib`) fitted exclusively on normal training records (`Machine failure == 0`).
  - Isolation Forest (`models/isolation_forest.joblib`) trained exclusively on scaled normal training records.
  - Evaluation population consists of held-out normal evaluation records (20% random split of normal data) combined with all 339 known machine-failure records (`Machine failure == 1`), totaling 2,272 evaluation records.
- **Frozen Artifact Execution:** The evaluation evaluates pre-trained, saved model artifacts without refitting or retraining.
- **Exact Reproduction Command:**

```powershell
.\.venv\Scripts\python.exe -m tests.manual_generate_results
```

---

## 3. Detector Evaluation Population and Performance

The frozen Isolation Forest was evaluated across all 2,272 held-out evaluation samples.

### Detector Population and Metrics Table

| Metric / Dimension | Value | Notes / Formulation |
| :--- | :--- | :--- |
| **Total Evaluation Records** | 2,272 | 1,933 held-out normal + 339 failure records |
| **All Detector-Predicted Anomalies** | 67 | Total records classified as anomalous (`label = -1`) |
| **Known Failures Among Anomalies (TP)** | 39 | True Positives (`Machine failure == 1` & predicted -1) |
| **Known Normal Records Among Anomalies (FP)** | 28 | False Positives (`Machine failure == 0` & predicted -1) |
| **Known Normal Records Correctly Classified (TN)** | 1,905 | True Negatives (`Machine failure == 0` & predicted 1) |
| **Known Failures Missed by Detector (FN)** | 300 | False Negatives (`Machine failure == 1` & predicted 1) |
| **Precision** | 58.21% (0.5821) | `TP / (TP + FP) = 39 / 67` |
| **Recall** | 11.50% (0.1150) | `TP / (TP + FN) = 39 / 339` |
| **F1 Score** | 19.21% (0.1921) | Harmonic mean of Precision and Recall |

### Confusion Matrix Breakdown

```
                     Predicted Normal (1)    Predicted Anomaly (-1)
Actual Normal (0)          1,905 (TN)               28 (FP)
Actual Failure (1)           300 (FN)               39 (TP)
```

### Explanatory Context
- **Predicted Anomaly vs. Known Failure:** An anomaly detection prediction is an unsupervised distributional judgment, not a direct failure classification. Of the 67 predicted anomalies, 39 coincided with actual machine failures, yielding a precision of 58.21%.
- **Limited Detector Recall:** The unsupervised detector captured 39 of the 339 total machine failures (11.50% recall). Limited recall shows that this frozen unsupervised detector did not detect every labelled failure in this dataset; this experiment does not establish the physical reason for each missed failure.
- **Impact on Downstream Explanations:** The counterfactual explanation pipeline is designed to explain anomalous decisions made by the detector. Consequently, downstream counterfactual explanations apply strictly to the 39 failures detected by the model.

---

## 4. Counterfactual Evaluation Population

The primary counterfactual evaluation denominator is strictly defined as:

> **Known machine-failure records that the frozen detector classified as anomalous.**

- **Primary Case Count:** Exactly **39 records**.
- **Scope Boundary:** It is methodologically incorrect to treat the denominator as all 339 known machine failures. The explanation pipeline explains *why an anomaly detector flagged a point*. Feeding undetected failures (points the detector considered normal) to the explanation engine would ask for counterfactuals on points that are already classified as normal. Furthermore, reporting success over all 339 failures would improperly conflate detector recall with counterfactual generation capability.

---

## 5. Counterfactual Results

The counterfactual search engine, feasibility checker, and plausibility module were run against all 39 primary failure cases.

### Counterfactual Performance Metrics Table

| Metric / Dimension | Value | Definition / Specification |
| :--- | :--- | :--- |
| **Primary Counterfactual Cases** | 39 | Evaluation denominator (TP cases) |
| **Counterfactuals Found** | 36 | Successfully transitioned detector label from -1 to 1 |
| **Counterfactuals Not Found** | 3 | Search budget exhausted without reaching normal boundary |
| **Counterfactual Found Rate** | **92.31%** (0.9231) | Conditional on detected known failures (`36 / 39`) |
| **Valid Counterfactuals** | 36 | 100.00% of found counterfactuals classified as normal |
| **Feasible Counterfactuals** | 36 | 100.00% of found counterfactuals respect domain bounds |
| **Plausible Counterfactuals** | 9 | Met Mahalanobis distance threshold against normal data |
| **Plausibility Rate** | **25.00%** (0.2500) | Conditional on counterfactuals found (`9 / 36`) |
| **Average Sparsity** | 1.00 | Average number of mutable features modified per CF |
| **Average Normalized Proximity** | 0.21 (0.2096) | Mean L1 distance scaled by feature range |
| **Average Generation Time** | 0.33 s (0.3263 s) | Mean runtime per case |
| **Mahalanobis Plausibility Threshold** | 3.17 (3.1706) | Empirically derived 95th percentile of normal data |
| **No-Counterfactual Row Indices** | `1954`, `1955`, `2166` | Evaluation indices where search budget was exceeded |

---

## 6. Feature-Change Findings

For all 36 successful counterfactuals, feature modifications were tracked:

| Feature Name | Modification Frequency | Role in Counterfactual Policy |
| :--- | :---: | :--- |
| **Torque [Nm]** | 26 | Mutable operational control parameter |
| **Rotational speed [rpm]** | 10 | Mutable operational control parameter |
| **Air temperature [K]** | 0 | Immutable ambient environmental condition |
| **Process temperature [K]** | 0 | Immutable physical lag state |
| **Tool wear [min]** | 0 | Immutable irreversible degradation state |

### Policy Enforcement
- **Mutable vs. Immutable Policy:** The explanation policy strictly enforced domain rules: `Air temperature [K]`, `Process temperature [K]`, and `Tool wear [min]` were treated as immutable. Only `Torque [Nm]` and `Rotational speed [rpm]` were allowed to change.
- **Sparsity:** Every found counterfactual required modifying only **1 feature** (`average_sparsity = 1.00`), highlighting minimal-intervention recourse paths.
- **Non-Causal Note:** These feature adjustments represent model-boundary transitions. They do not constitute mechanical or causal root causes of machine failure.

---

## 7. Interpretation

> *Among the known machine-failure records detected as anomalous by the frozen Isolation Forest baseline, the system generated feasible counterfactual explanations for the reported proportion of cases (92.31%). A smaller subset (25.00%) also met the normal-distribution plausibility criterion.*

### Key Insights: Validity, Feasibility, and Plausibility
- **Validity & Feasibility vs. Plausibility:** A counterfactual is **valid** if the detector assigns it a normal prediction (`+1`), and **feasible** if its feature values comply with the configured mutable-feature policy and empirical feature bounds. However, feasibility and validity do not guarantee physical safety or that the point resembles typical operational history.
- **Role of the Plausibility Metric:** The Mahalanobis plausibility check evaluates whether the counterfactual lies within the multivariate distribution of historical normal operations, accounting for covariance between rotational speed and torque. 
- **Transparent Reporting:** The system explicitly distinguishes between "Found and plausible" (9 cases) and "Found but implausible" (27 cases). Implausible cases often land in regions of the feature space that are outside, or in low-density regions of, the observed normal training-data distribution, even if permitted by the Isolation Forest decision boundary. By reporting these openly, the system avoids misleading operators with ungrounded recommendations.

---

## 8. Limitations and Responsible-Use Boundaries

1. **Not Causal Root-Cause Conclusions:** Counterfactual explanations indicate minimal changes that alter the model's classification score. They do not identify the root failure mechanism or the physical sequence of degradation.
2. **Not Certified-Safe Operational Actions:** Proposed adjustments to torque or rotational speed must not be executed automatically on physical machinery. They require engineering validation.
3. **No Guarantee of Repair or Failure Prevention:** Modifying an operational parameter does not eliminate internal mechanical damage (such as tool wear, thermal fatigue, or bearing strain).
4. **Limited Detector Recall:** The unsupervised Isolation Forest detects 11.50% of machine failures. It is not an exhaustive safety monitor.
5. **Dependence on Configured Policy:** Results depend directly on configured feature bounds, mutable feature rules, and the chosen 95th-percentile Mahalanobis distance threshold.
6. **Empirical Bounds vs. Engineering Envelopes:** Empirical min-max bounds derived from the training partition reflect observed historical ranges, not physical structural limits.

---

## 9. Generated Evidence

All quantitative findings and diagnostic plots are automatically saved in the `results/` directory:

### Metrics (`results/metrics/`)
- `detector_metrics.json`: Full population counts, precision, recall, F1, and confusion matrix for the detector evaluation.
- `counterfactual_metrics.json`: Summary of search rates, validity, feasibility, plausibility rates, sparsity, proximity, generation time, and limitation statements.

### Explanations (`results/explanations/`)
- `counterfactual_case_results.csv`: Row-by-row accounting for all 39 primary failure cases. Contains row indices, anomaly scores, search status, failure reasons, candidate evaluation counts, plausibility distances, and full original vs. counterfactual feature states.

### Visualizations (`results/plots/`)
- `detector_confusion_matrix.png`: Visualizes true negatives, false positives, false negatives, and true positives across the evaluation population.
- `counterfactual_outcome_breakdown.png`: Bar chart displaying the exact distribution across the three outcome classes: `Found and plausible` (9), `Found but implausible` (27), and `Not found` (3).
- `changed_feature_frequency.png`: Bar chart displaying the frequency of feature changes (`Torque [Nm]`: 26, `Rotational speed [rpm]`: 10).
- `plausibility_distance_distribution.png`: Histogram showing Mahalanobis distances of valid counterfactuals relative to the red dashed empirical cutoff threshold (`3.17`).
- `generation_time_distribution.png`: Histogram of generation runtimes per case, showing the observed generation-time distribution in this evaluation run.
