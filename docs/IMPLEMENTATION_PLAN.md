# Implementation Plan

## Project

**Automated Anomaly Explanation with Counterfactuals**

## Purpose

This plan defines the controlled build sequence for the hackathon prototype.

The project goal is:

> Given an anomalous industrial machine observation, find the smallest feasible change that moves it into the anomaly detector’s learned normal region and explain that recommendation clearly.

Each phase must be completed, tested, reviewed, and committed before moving to the next phase.

## Working Method

```text
One small task
   ↓
Clear requirement and allowed files
   ↓
Implementation with assisted coding where appropriate
   ↓
Run and inspect real output
   ↓
Review result
   ↓
Git commit
   ↓
Next small task
```

The project must not skip directly to counterfactuals or UI before the anomaly detector works.

## Completed Work

| Status | Phase | Completed Output |
|---|---|---|
| ✅ | Project scaffold | Project folders, placeholder source files, and documentation structure |
| ✅ | Product requirements | `docs/PRD.md` |
| ✅ | Software requirements | `docs/SRS.md` |
| ✅ | Dataset setup | AI4I 2020 CSV stored in `data/raw/` |
| ✅ | Raw-data protection | `.gitignore` excludes the raw CSV from Git |
| ✅ | Data exploration | `notebooks/01_data_understanding_eda.ipynb` created, run, and reviewed |
| ✅ | Data-quality verification | Confirmed no missing values, duplicate rows, or duplicate column names |
| ✅ | Data dictionary | `docs/DATA_DICTIONARY.md` |
| ✅ | Feature policy | `config/features.yaml` |
| ✅ | Version control | Git repository initialized and initial project checkpoint created |

## Confirmed Baseline Decisions

### Detector Inputs

The first Isolation Forest model will use:

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

### Excluded Columns

The following columns must not be detector inputs:

```text
UDI
Product ID
Type
Machine failure
TWF
HDF
PWF
OSF
RNF
```

### Training Policy

```text
Train using known-normal records only:
Machine failure = 0

Split known-normal records:
80% training
20% held-out normal evaluation

Keep all known machine-failure records outside model training.
```

### Counterfactual Policy: Version 1

Counterfactual search may change only:

```text
Rotational speed [rpm]
Torque [Nm]
```

All other fields remain unchanged.

Observed dataset minimum and maximum values are initial data-based search bounds. They are not certified physical safety limits.

## Remaining Implementation Roadmap

| Status | Phase | Main Output | Completion Condition |
|---|---|---|---|
| ⏭️ Next | Model configuration | `config/model_config.yaml` | Training split, model parameters, threshold policy, and evaluation settings defined |
| ⬜ | Data loader | `src/data_loader.py` | CSV loads, validates expected schema, and reports actionable errors |
| ⬜ | Preprocessing | `src/preprocessing.py` | Correct features selected, scaler fitted on training data only, transformations reproducible |
| ⬜ | Detector | `src/detector.py` | Isolation Forest trains, scores samples, and returns normal/anomaly decisions |
| ⬜ | Detector evaluation | Evaluation output | Detector results measured against known labels without label leakage |
| ⬜ | Counterfactual specification | `docs/COUNTERFACTUAL_SPEC.md` | Validity, sparsity, proximity, bounds, and search limits defined |
| ⬜ | Counterfactual engine | `src/counterfactual.py` | Candidate counterfactuals generated and verified by the detector |
| ⬜ | Feasibility layer | `src/feasibility.py` | Range, mutability, and plausibility checks enforced |
| ⬜ | Explanation generator | `src/explanation.py` | Before/after values converted into honest engineer-friendly explanations |
| ⬜ | Pipeline | `src/pipeline.py` | One end-to-end `analyze(sample)` workflow works |
| ⬜ | Counterfactual evaluation | `src/evaluator.py` | Validity, sparsity, proximity, plausibility, and time measured |
| ⬜ | Testing | `tests/` | Normal, anomaly, invalid-input, and no-solution cases tested |
| ⬜ | Demo interface | `app/app.py` | Streamlit app displays actual pipeline results |
| ⬜ | Final documentation | README and remaining documents | Documentation reflects actual implementation and measured results |
| ⬜ | Final presentation | Presentation/report | Demo story, limitations, and evidence prepared |

## Phase Rules

1. Do not create ML code before the relevant configuration is approved.
2. Do not use known failure labels as detector input features.
3. Do not claim a counterfactual is valid until the trained detector classifies it as normal.
4. Do not allow immutable features to change.
5. Do not present observed data ranges as certified engineering safety limits.
6. Do not fabricate metrics, explanations, or results.
7. Keep each coding task narrow and reviewable.
8. Commit each completed and verified phase to Git.
9. Keep core ML logic in `src/`, not in the Streamlit UI.
10. Prefer a working baseline over advanced but incomplete features.

## Immediate Next Step

Create `config/model_config.yaml`.

This configuration will define:

- normal-data training split;
- reproducible random state;
- Isolation Forest parameters;
- anomaly-decision policy;
- evaluation settings;
- initial counterfactual-search limits.

Only after this configuration is complete should implementation begin in `src/data_loader.py` and `src/preprocessing.py`.