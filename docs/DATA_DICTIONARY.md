# Data Dictionary

## 1. Dataset Overview

**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Raw file:** `data/raw/ai4i2020.csv`  
**Records:** 10,000  
**Columns:** 14  
**Data quality:** No missing values, no duplicate rows, and no duplicate column names were found during initial EDA.

This dataset represents industrial machine operating records. It contains machine identifiers, product-type information, five operational measurements, one known machine-failure label, and five specific failure-mode labels.

The project uses this dataset to train an unsupervised anomaly detector and generate counterfactual explanations for unusual machine states.

## 2. Important Interpretation Rule

`Machine failure` is a known dataset failure label. It is used for evaluation and for selecting normal operational records during training.

It is **not** an input feature and it is **not** the Isolation Forest prediction. The detector will produce its own anomaly score and normal/anomaly decision.

Likewise, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are known failure-mode labels. They are excluded from model inputs because using them would directly leak outcome information into the detector.

## 3. Data Quality Findings

| Check | Result |
|---|---|
| Missing values | 0 across all columns |
| Duplicate rows | 0 |
| Duplicate column names | 0 |
| Dataset size | 10,000 rows × 14 columns |
| Known normal records | 9,661 (`Machine failure = 0`) |
| Known machine-failure records | 339 (`Machine failure = 1`) |
| Failure proportion | 3.39% |

The dataset is clean enough for the first baseline. No missing-value imputation is expected for the supplied CSV, but preprocessing code must still validate future input before inference.

## 4. Column-Level Dictionary

| Column | Data Type | Observed Values / Range | Role | Baseline Model Use | Counterfactual Policy |
|---|---|---:|---|---|---|
| `UDI` | Integer | 1 to 10,000 | Row identifier | Exclude | Immutable; never change |
| `Product ID` | String | 10,000 unique values | Product identifier | Exclude | Immutable; never change |
| `Type` | Categorical | `L`, `M`, `H` | Product-type metadata | Exclude in baseline | Immutable; never change |
| `Air temperature [K]` | Float | 295.3 to 304.5 K | Operational measurement | Include | Immutable in baseline counterfactuals |
| `Process temperature [K]` | Float | 305.7 to 313.8 K | Operational measurement | Include | Immutable in baseline counterfactuals |
| `Rotational speed [rpm]` | Integer | 1,168 to 2,886 rpm | Operational measurement | Include | Mutable in baseline counterfactuals |
| `Torque [Nm]` | Float | 3.8 to 76.6 Nm | Operational measurement | Include | Mutable in baseline counterfactuals |
| `Tool wear [min]` | Integer | 0 to 253 min | Operational measurement | Include | Immutable in baseline; maintenance-only action in a future version |
| `Machine failure` | Binary integer | 0 or 1 | Known failure label | Exclude | Never change |
| `TWF` | Binary integer | 46 positive records | Tool Wear Failure label | Exclude | Never change |
| `HDF` | Binary integer | 115 positive records | Heat Dissipation Failure label | Exclude | Never change |
| `PWF` | Binary integer | 95 positive records | Power Failure label | Exclude | Never change |
| `OSF` | Binary integer | 98 positive records | Overstrain Failure label | Exclude | Never change |
| `RNF` | Binary integer | 19 positive records | Random Failure label | Exclude | Never change |

## 5. Baseline Detector Features

The first Isolation Forest baseline will use only these five continuous operational features:

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

This gives a small, interpretable baseline and avoids identifier leakage, target leakage, and unnecessary categorical encoding.

`Type` will remain excluded in version 1. A later experiment may compare this baseline with an alternative that one-hot encodes `Type` as an immutable product-category feature.

## 6. Training and Evaluation Data Policy

The detector must be trained only on records where:

```text
Machine failure = 0
```

Before training, those 9,661 known-normal records must be split reproducibly:

```text
80% normal training data
20% held-out normal evaluation data
```

All 339 known machine-failure records must remain outside model training.

Evaluation should use:

```text
Held-out known-normal records
+
All known machine-failure records
```

This allows the project to compare the unsupervised detector’s predictions with known dataset failure labels without leaking those labels into training.

## 7. Counterfactual Feature Rules for Version 1

The baseline counterfactual engine will search for the smallest feasible changes using only:

```text
Rotational speed [rpm]
Torque [Nm]
```

These are treated as the initial adjustable operational settings.

The following features remain unchanged in version 1:

```text
UDI
Product ID
Type
Air temperature [K]
Process temperature [K]
Tool wear [min]
Machine failure
TWF
HDF
PWF
OSF
RNF
```

`Tool wear [min]` must not be reduced mathematically during a real-time recommendation. Tool wear is cumulative. A future version may model a separate discrete maintenance action such as “replace the tool,” but that is outside the first counterfactual-search baseline.

## 8. Observed Bounds and Feasibility Warning

The following values are the minimum and maximum measurements observed in this dataset:

| Feature | Observed Minimum | Observed Maximum |
|---|---:|---:|
| Air temperature [K] | 295.3 | 304.5 |
| Process temperature [K] | 305.7 | 313.8 |
| Rotational speed [rpm] | 1,168 | 2,886 |
| Torque [Nm] | 3.8 | 76.6 |
| Tool wear [min] | 0 | 253 |

These are **empirical data bounds**, not proven machine-safety limits.

For the hackathon baseline, counterfactual candidates must remain within these observed bounds. The project must describe this honestly as a data-based feasibility constraint, not as a certified engineering safety specification.

## 9. Relationships Relevant to Plausibility

EDA found two important correlations:

- Rotational speed and torque have a strong negative correlation of approximately `-0.875`.
- Air temperature and process temperature have a strong positive correlation of approximately `+0.876`.

These relationships are useful for plausibility checking. A counterfactual should not produce a feature combination far outside the learned normal operating pattern.

Correlation alone does not prove a physical law or causal relationship. The plausibility layer will therefore use these patterns as statistical evidence, not as a complete engineering model.

## 10. Failure-Label Notes

The five specific failure-mode labels do not perfectly map one-to-one to `Machine failure`.

EDA found:

- 24 records with more than one specific failure-mode label.
- 9 records where `Machine failure = 1` but all specific failure-mode labels are 0.
- 18 records where `Machine failure = 0` but `RNF = 1`.

For this reason, the baseline uses `Machine failure` as the primary known label for training-pool selection and evaluation. The individual failure-mode columns may be used later for exploratory analysis but must remain excluded from detector inputs.

## 11. Decisions Locked for Baseline Version 1

The following decisions apply until an explicit project change is approved:

- Use the five continuous operational features as Isolation Forest inputs.
- Exclude identifiers, `Type`, and all labels from model inputs.
- Train Isolation Forest on known-normal records only.
- Use an 80/20 split of normal records for training and held-out normal evaluation.
- Keep all known machine-failure records out of detector training.
- Permit counterfactual changes only to rotational speed and torque.
- Treat observed data ranges as initial search bounds, not certified safety limits.
- Treat tool wear as immutable during real-time counterfactual search.