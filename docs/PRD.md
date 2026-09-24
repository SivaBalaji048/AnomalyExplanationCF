# Product Requirements Document (PRD)

Product and problem definition placeholder.
# Product Requirements Document (PRD)

## Project Title

Automated Anomaly Explanation with Counterfactuals

## 1. Product Summary

This project is an industrial predictive-maintenance decision-support prototype.

Traditional anomaly detection systems usually return only an anomaly score or an anomaly label. That tells an engineer that a machine observation is unusual, but it does not explain what made it unusual or what practical change could move the machine back toward normal operation.

This product will detect anomalous machine observations and generate a verified counterfactual explanation. A counterfactual is a minimally changed version of the anomalous observation that the anomaly detector classifies as normal.

The product’s central promise is:

> Detect an anomalous machine state, identify the smallest realistic change that returns it to the learned normal operating region, verify that the recommendation is feasible, and explain it clearly.

## 2. Problem Statement

Engineers need answers to the following questions when a machine state is detected as anomalous:

- Why was this observation considered unusual?
- Which operational features contribute to the abnormal state?
- What is the smallest change that could make the observation normal according to the model?
- Is the suggested change within valid, realistic, and allowed operating limits?
- Can the result be communicated as an actionable recommendation rather than as a raw ML score?

A raw anomaly score alone does not answer these questions. The project therefore treats anomaly explanation as a constrained counterfactual-generation problem.

## 3. Target Users

Primary users:

- Maintenance engineers who need practical guidance from machine-condition data.
- Operations engineers who need to understand abnormal operating states.
- Hackathon judges or technical reviewers evaluating explainable AI functionality.

Secondary users:

- Data scientists who want a transparent baseline for anomaly explanation.
- Students or developers learning the relationship between anomaly detection and counterfactual explanations.

## 4. Product Goal

Build a complete, reliable end-to-end prototype that accepts an industrial machine observation and returns one of two outcomes:

1. If the observation is normal, clearly report that it is normal and provide its anomaly score.

2. If the observation is anomalous, provide:
   - anomaly score and anomaly status;
   - one or more candidate counterfactuals;
   - a ranked best counterfactual;
   - changed features with before/after values;
   - verification that the counterfactual changes the detector result from anomaly to normal;
   - feasibility and plausibility status;
   - a concise human-readable engineering explanation.

## 5. Core Product Flow

```text
Machine observation
        ↓
Data validation and preprocessing
        ↓
Isolation Forest anomaly detector
        ↓
Normal or anomalous decision
        ↓
If anomalous: constrained counterfactual search
        ↓
Hard feasibility checks
        ↓
Plausibility checks
        ↓
Ranking of valid candidates
        ↓
Human-readable explanation
        ↓
Evaluation metrics and demo presentation
```

## 6. Primary Dataset and Domain

The hackathon implementation will use the AI4I 2020 Predictive Maintenance Dataset unless the project owner explicitly changes that decision.

The project should be framed around industrial machine operation and predictive maintenance. It must not become a generic anomaly explanation framework across unrelated domains during the initial hackathon build.

Dataset-specific facts, including exact column names, units, data ranges, labels, feature meanings, and mutability rules, must be derived from the actual dataset during exploratory data analysis. They must not be guessed or hard-coded in advance.

## 7. Functional Goals

The product must:

- Load and validate the raw dataset.
- Identify usable model features after inspecting the data.
- Preserve identifiers and known labels for traceability and evaluation where appropriate.
- Train an Isolation Forest anomaly detector primarily on normal operational data.
- Score and classify a new machine observation as normal or anomalous.
- Generate candidate counterfactuals for anomalous observations.
- Change only features explicitly marked as mutable.
- Keep every changed value inside its valid range.
- Reject invalid, missing, or infeasible counterfactual candidates.
- Apply a lightweight plausibility check based on the learned normal-data region.
- Prefer counterfactuals that change fewer features and make smaller changes.
- Confirm that every returned counterfactual is truly classified as normal by the same trained detector.
- Generate an engineer-friendly explanation supported by the actual counterfactual values.
- Measure quality and time cost over a set of anomalous observations.
- Present the final workflow in a simple Streamlit demo.

## 8. Product Success Criteria

The hackathon prototype is successful when it can demonstrate the following complete workflow:

- A real machine observation is loaded and preprocessed correctly.
- The trained detector produces an anomaly score and decision.
- At least one anomalous observation receives a valid counterfactual.
- The counterfactual changes the detector’s output from anomaly to normal.
- The proposed change respects feature mutability and observed or defined valid ranges.
- The candidate passes the defined plausibility check.
- The explanation states only changes actually present in the counterfactual.
- Results are evaluated on approximately 20 to 30 anomalous observations where the dataset supports this.
- The system reports validity, sparsity, proximity, plausibility, and generation time.
- A user can understand the before/after recommendation in the demo without reading source code.

## 9. Required Quality Measures

The project must measure these counterfactual-quality dimensions:

### Validity

A counterfactual is valid only if the trained anomaly detector classifies the modified observation as normal.

### Sparsity

The number of features changed between the original anomaly and the counterfactual. Fewer changed features are preferred.

### Proximity

The size of the change from the original observation to the counterfactual. Smaller normalized changes are preferred.

### Plausibility

Whether the counterfactual is consistent with valid feature constraints and sufficiently close to the learned normal-data region according to the defined plausibility method.

### Generation Time

The time required to produce and rank a counterfactual recommendation.

## 10. Scope for the Hackathon Version

In scope:

- One industrial predictive-maintenance dataset.
- One primary anomaly detector: Isolation Forest.
- Data loading, validation, exploratory data analysis, and preprocessing.
- A custom constrained counterfactual-search baseline.
- Hard feasibility checks.
- A lightweight statistical or distance-based plausibility check.
- Candidate ranking.
- Template-based human-readable explanations.
- Counterfactual evaluation.
- A small Streamlit demonstration interface.
- Essential documentation and final presentation material.

## 11. Out of Scope

The initial hackathon version must not include the following unless the project owner explicitly expands the scope:

- Multiple anomaly-detector comparison suites.
- Deep-learning anomaly detectors or autoencoders.
- Complex causal inference.
- Custom neural counterfactual models.
- LLM-dependent explanation logic.
- A database, user accounts, authentication, or authorization.
- Microservices, Kubernetes, cloud infrastructure, or production deployment.
- A complex React or enterprise frontend.
- Multiple unrelated datasets or domains.
- Hard-coded fake results, fabricated evaluation metrics, or unsupported engineering claims.

## 12. Product Constraints and Principles

- The project must prioritize a working vertical slice over advanced but incomplete components.
- Core ML logic must remain outside the UI layer.
- The counterfactual engine must query the detector through a clean, deterministic interface.
- A returned recommendation must never claim validity unless it truly flips the detector output to normal.
- A returned recommendation must never claim plausibility unless it passes the implemented plausibility check.
- Immutable features must never be changed.
- Dataset-specific ranges must be discovered from the actual data and documented.
- The preprocessing transformation must be preserved so results can be converted back to understandable original units.
- The product must document limitations honestly.
- The implementation must preserve the agreed architecture unless a technical reason and explicit approval justify a change.

## 13. User-Facing Output Example

The wording below is illustrative only. Values must always come from an actual model result.

```text
ANOMALY DETECTED

Anomaly score: <actual score>

Recommended adjustment:
<Feature name>: <original value> → <counterfactual value>

Required change:
<actual direction and magnitude>

Features changed: <actual count>

Detector status after change: NORMAL
Feasibility check: PASSED
Plausibility check: PASSED

Explanation:
The machine observation is unusual under the learned normal operating
pattern. Reducing/increasing <feature> by <amount> is the smallest
verified feasible change found by the system that moves this observation
back into the normal region.
```

## 14. Definition of Done

The hackathon version is complete when all of the following are true:

- Dataset is loaded and understood.
- Preprocessing is reproducible.
- Isolation Forest is trained and evaluated.
- Anomaly scores and decisions are available.
- Counterfactuals are generated for anomalous points.
- Counterfactuals are verified to flip anomaly to normal.
- Range, mutability, and plausibility constraints are enforced.
- Candidates are ranked.
- Human-readable explanations are generated from actual outputs.
- Quality metrics and generation time are measured.
- The Streamlit demo runs end-to-end.
- Documentation reflects real implementation choices and measured results.
- The project can be clearly explained through its central value proposition.

## 15. North Star

> We do not only tell an engineer that a machine state is anomalous. We find the smallest realistic change that would move it back into the learned normal operating region and explain that change clearly.