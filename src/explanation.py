"""Human-readable explanation module.

Responsibilities:
- convert counterfactual results into human-readable explanations
"""

import re
from typing import Dict, Any, List

class ExplanationGenerator:
    """Generates human-readable explanations from anomaly detection pipeline results."""

    def __init__(self):
        pass

    def _format_label(self, label: int) -> str:
        if label == -1:
            return "anomalous"
        elif label == 1:
            return "normal"
        else:
            raise ValueError(f"Unexpected label received: {label}. Expected 1 or -1.")

    def _format_score(self, score: float) -> str:
        if isinstance(score, float):
            formatted = f"{score:.7f}".rstrip('0')
            if formatted.endswith('.'):
                formatted += '0'
            return formatted
        return str(score)
        
    def _extract_feature_and_unit(self, feature_name: str) -> tuple[str, str]:
        """Extracts feature name and unit (if present in brackets)."""
        match = re.search(r"^(.*?)(?:\s*\[(.*?)\])?$", feature_name)
        if match:
            name = match.group(1).strip()
            unit = match.group(2).strip() if match.group(2) else ""
            return name, unit
        return feature_name.strip(), ""

    def _format_value(self, value: float) -> str:
        """Sensible numeric formatting without excessive decimals."""
        if isinstance(value, float):
            # Format to 4 significant digits
            formatted = f"{value:.4g}"
            # If it's effectively an integer but was a float, append .0 to match example
            if '.' not in formatted and 'e' not in formatted.lower():
                formatted += ".0"
            return formatted
        return str(value)

    def _generate_changed_features_text(self, changed_features: List[Dict[str, Any]]) -> str:
        if not changed_features:
            return "No features were changed."
            
        parts = []
        for feat in changed_features:
            raw_name = feat.get("feature", "Unknown")
            orig_val = feat.get("original_value")
            cf_val = feat.get("counterfactual_value")
            delta = feat.get("delta")
            
            name, unit = self._extract_feature_and_unit(raw_name)
            unit_str = f" {unit}" if unit else ""
            
            orig_str = f"{self._format_value(orig_val)}{unit_str}" if orig_val is not None else "Unknown"
            cf_str = f"{self._format_value(cf_val)}{unit_str}" if cf_val is not None else "Unknown"
            
            delta_str = ""
            if delta is not None:
                sign = "+" if delta > 0 else ""
                delta_str = f" (delta {sign}{self._format_value(delta)}{unit_str})"
                
            parts.append(f"{name} changed from {orig_str} to {cf_str}{delta_str}")
            
        if len(parts) == 1:
            return parts[0] + "."
        elif len(parts) == 2:
            return parts[0] + " and " + parts[1] + "."
        else:
            return ", ".join(parts[:-1]) + ", and " + parts[-1] + "."

    def generate(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a structured explanation based on the pipeline result."""
        
        # 1. Input Validation
        required_base = ["found", "reason", "original_label"]
        for key in required_base:
            if key not in pipeline_result:
                raise ValueError(f"Missing required key in pipeline_result: {key}")
                
        found = pipeline_result["found"]
        reason = pipeline_result["reason"]
        original_label_raw = pipeline_result["original_label"]
        original_label = self._format_label(original_label_raw)
        
        if found:
            required_cf = ["counterfactual_label", "changed_features", "feasibility_status", "plausibility_status"]
            for key in required_cf:
                if key not in pipeline_result:
                    raise ValueError(f"Missing required counterfactual key in pipeline_result: {key}")
                    
            cf_label_raw = pipeline_result["counterfactual_label"]
            cf_label = self._format_label(cf_label_raw)
            changed_features = pipeline_result["changed_features"]
            feasibility_status = pipeline_result["feasibility_status"]
            plausibility_status = pipeline_result["plausibility_status"]

        # Initialize outputs
        summary = ""
        anomaly_status = f"The observed machine state was classified as {original_label} by the trained anomaly detector."
        cf_summary = ""
        detector_transition = ""
        feas_summary = ""
        plaus_summary = ""
        limitations = [
            "This is a model-based counterfactual, not a causal explanation.",
            "The counterfactual does not establish that the changed feature caused the anomaly.",
            "The result is not a certified physical safety or operating recommendation."
        ]

        # Process cases
        if not found:
            if reason == "original_state_is_already_normal":
                summary = "The observed machine state was classified as normal by the trained anomaly detector. No counterfactual explanation was required."
                anomaly_status = "The observed machine state was classified as normal by the trained anomaly detector."
            else:
                summary = "No valid counterfactual was found within the configured search budget."
                anomaly_status = f"The observed machine state was classified as anomalous by the trained anomaly detector."
                limitations.append("Failure to find a counterfactual within the configured search budget does not prove that no counterfactual exists.")
        else:
            summary = "A counterfactual state was found that is classified as normal."
            anomaly_status = f"The observed machine state was classified as anomalous by the trained Isolation Forest."
            
            cf_summary = f"A counterfactual was found by {self._generate_changed_features_text(changed_features).replace('changed from', 'changing')}".replace(' changing', ' changing ') 
            # Fix grammar logic: "changing Torque from X to Y"
            cf_summary = "A counterfactual was found by "
            
            if not changed_features:
                cf_summary += "making no feature changes."
            else:
                parts = []
                for feat in changed_features:
                    raw_name = feat.get("feature", "Unknown")
                    orig_val = feat.get("original_value")
                    cf_val = feat.get("counterfactual_value")
                    
                    name, unit = self._extract_feature_and_unit(raw_name)
                    unit_str = f" {unit}" if unit else ""
                    
                    orig_str = f"{self._format_value(orig_val)}{unit_str}" if orig_val is not None else "Unknown"
                    cf_str = f"{self._format_value(cf_val)}{unit_str}" if cf_val is not None else "Unknown"
                    
                    parts.append(f"changing {name} from {orig_str} to {cf_str}")
                    
                if len(parts) == 1:
                    cf_summary += parts[0] + " while keeping the other detector inputs unchanged."
                elif len(parts) == 2:
                    cf_summary += parts[0] + " and " + parts[1] + " while keeping the other detector inputs unchanged."
                else:
                    cf_summary += ", ".join(parts[:-1]) + ", and " + parts[-1] + " while keeping the other detector inputs unchanged."
            
            # Detector transition
            detector_transition = "The original state was classified as anomalous by the trained detector. The counterfactual state was classified as normal."
            if "original_score" in pipeline_result and "counterfactual_score" in pipeline_result:
                detector_transition += f" Detector score changed from {self._format_score(pipeline_result['original_score'])} to {self._format_score(pipeline_result['counterfactual_score'])}."
            detector_transition += " The proposed counterfactual moved the state across the trained detector's decision boundary."

            # Feasibility
            if feasibility_status == "feasible":
                feas_summary = "The proposed counterfactual satisfies the configured feasibility constraints, including allowed mutability, bounds, and step alignment."
            elif feasibility_status == "infeasible":
                feas_summary = "The proposed counterfactual does not satisfy all configured feasibility constraints."
            elif feasibility_status == "pending":
                feas_summary = "Feasibility has not yet been evaluated."
            elif feasibility_status == "missing":
                feas_summary = "Feasibility information is missing."
            else:
                feas_summary = f"Feasibility status is {feasibility_status}."
                
            # Plausibility
            if plausibility_status == "plausible":
                plaus_summary = "The proposed state is within the learned statistical plausibility threshold derived from the normal training population."
            elif plausibility_status == "implausible":
                plaus_summary = "The proposed state is outside the learned statistical plausibility threshold derived from the normal training population."
                limitations.append("The candidate crossed the detector boundary but did not satisfy the current statistical plausibility criterion.")
            elif plausibility_status == "pending":
                plaus_summary = "Plausibility has not yet been evaluated."
            elif plausibility_status == "missing":
                plaus_summary = "Plausibility information is missing."
            else:
                plaus_summary = f"Plausibility status is {plausibility_status}."

        # Construct full explanation
        full_text_parts = []
        if not found:
            full_text_parts.append(summary)
            if limitations:
                full_text_parts.append(" ".join(limitations))
        else:
            full_text_parts.append("Anomaly detected.")
            full_text_parts.append(anomaly_status)
            full_text_parts.append(cf_summary)
            if detector_transition:
                # Omit the first sentence of transition to avoid repeating anomaly status exactly
                # Actually, the spec example is:
                # "Anomaly detected. The observed machine state was classified as anomalous by the trained Isolation Forest. A counterfactual was found by changing Torque from 4.6 Nm to 17.0 Nm while keeping the other detector inputs unchanged. The resulting state was classified as normal by the detector. The proposed change satisfies the configured feasibility constraints. However, the candidate is outside the learned statistical plausibility threshold. This is a model-based counterfactual and does not establish causality or physical safety."
                # We can adjust it based on components.
                
                # Simplified transition string for the full text, as anomaly_status already says it's anomalous
                transition_short = "The resulting state was classified as normal by the detector."
                full_text_parts.append(transition_short)
                if "original_score" in pipeline_result and "counterfactual_score" in pipeline_result:
                    full_text_parts.append(f"Detector score changed from {self._format_score(pipeline_result['original_score'])} to {self._format_score(pipeline_result['counterfactual_score'])}.")

            if feas_summary:
                if feasibility_status == "feasible":
                    full_text_parts.append("The proposed change satisfies the configured feasibility constraints.")
                else:
                    full_text_parts.append(feas_summary)
                    
            if plaus_summary:
                if plausibility_status == "implausible":
                    full_text_parts.append("However, the candidate is outside the learned statistical plausibility threshold.")
                else:
                    full_text_parts.append(plaus_summary)
                    
            # Add general limitations at the end
            full_text_parts.append("This is a model-based counterfactual and does not establish causality or physical safety.")

        full_explanation = " ".join(full_text_parts)

        return {
            "summary": summary,
            "anomaly_status": anomaly_status,
            "counterfactual_summary": cf_summary,
            "detector_transition": detector_transition,
            "feasibility_summary": feas_summary,
            "plausibility_summary": plaus_summary,
            "limitations": limitations,
            "full_explanation": full_explanation
        }
