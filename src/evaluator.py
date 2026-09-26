"""Evaluation module.

Responsibilities:
- validity
- sparsity
- proximity
- plausibility
- generation time metrics
"""

from typing import List, Dict, Any, Optional
import sklearn.metrics
import numpy as np

class Evaluator:
    """Evaluator for anomaly detection and counterfactual generation."""

    def evaluate_detector(self, y_true, y_pred) -> Dict[str, Any]:
        """
        Evaluate detector predictions against known labels.
        
        Args:
            y_true: True labels (1 = machine failure, 0 = normal)
            y_pred: Detector predictions (-1 = predicted anomaly, 1 = predicted normal)
            
        Returns:
            Dict containing precision, recall, f1_score, and confusion_matrix.
        """
        # Convert inputs to numpy arrays for safety and map detector labels
        y_true_mapped = np.array(y_true)
        # Map detector predictions: -1 -> 1 (predicted failure), 1 -> 0 (predicted normal)
        y_pred_mapped = np.where(np.array(y_pred) == -1, 1, 0)
        
        # Calculate metrics using pos_label=1
        precision = sklearn.metrics.precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
        recall = sklearn.metrics.recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
        f1 = sklearn.metrics.f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
        cm = sklearn.metrics.confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
        }

    def evaluate_counterfactual(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single counterfactual pipeline result.
        """
        # Input validation
        if not isinstance(pipeline_result, dict):
            raise ValueError("pipeline_result must be a dictionary")
        
        required_keys = ["found", "reason", "original_label"]
        for k in required_keys:
            if k not in pipeline_result:
                raise ValueError(f"Missing required key in pipeline_result: {k}")
                
        found = bool(pipeline_result["found"])
        reason = str(pipeline_result["reason"])
        original_label = int(pipeline_result["original_label"])
        
        # Initialize outputs
        valid = False
        cf_label = None
        sparsity = None
        norm_prox = None
        feasible = None
        feas_status = None
        plausible = None
        plaus_status = None
        plaus_dist = None
        gen_time = pipeline_result.get("generation_time_seconds")
        
        if found:
            # Need counterfactual label to check validity
            if "counterfactual_label" not in pipeline_result:
                raise ValueError("Missing 'counterfactual_label' for a found counterfactual.")
            cf_label = int(pipeline_result["counterfactual_label"])
            valid = (cf_label == 1)
            
            # Sparsity
            if "sparsity" in pipeline_result:
                sparsity = int(pipeline_result["sparsity"])
            elif "changed_features" in pipeline_result:
                sparsity = len(pipeline_result["changed_features"])
                
            # Normalized proximity
            if "normalized_proximity" in pipeline_result:
                norm_prox = float(pipeline_result["normalized_proximity"])
                
            # Feasibility
            feas_status = pipeline_result.get("feasibility_status")
            if feas_status == "feasible":
                feasible = True
            elif feas_status == "infeasible":
                feasible = False
                
            # Plausibility
            plaus_status = pipeline_result.get("plausibility_status")
            if plaus_status == "plausible":
                plausible = True
            elif plaus_status == "implausible":
                plausible = False
                
            if "plausibility_distance" in pipeline_result:
                plaus_dist = float(pipeline_result["plausibility_distance"])
                
        return {
            "found": found,
            "valid": valid,
            "reason": reason,
            "original_label": original_label,
            "counterfactual_label": cf_label,
            "sparsity": sparsity,
            "normalized_proximity": norm_prox,
            "feasible": feasible,
            "feasibility_status": feas_status,
            "plausible": plausible,
            "plausibility_status": plaus_status,
            "plausibility_distance": plaus_dist,
            "generation_time_seconds": gen_time
        }

    def evaluate_counterfactual_batch(self, pipeline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple pipeline results.
        """
        if not isinstance(pipeline_results, list):
            raise ValueError("pipeline_results must be a list of dictionaries")
            
        total_cases = len(pipeline_results)
        anomalies_analyzed = 0
        counterfactuals_found = 0
        valid_counterfactuals = 0
        feasible_counterfactuals = 0
        plausible_counterfactuals = 0
        
        sparsity_values = []
        proximity_values = []
        generation_times = []
        
        for res in pipeline_results:
            eval_res = self.evaluate_counterfactual(res)
            
            if eval_res["original_label"] == -1:
                anomalies_analyzed += 1
                
            if eval_res["found"]:
                counterfactuals_found += 1
                
                if eval_res["valid"]:
                    valid_counterfactuals += 1
                    
                if eval_res["feasible"] is True:
                    feasible_counterfactuals += 1
                    
                if eval_res["plausible"] is True:
                    plausible_counterfactuals += 1
                    
                if eval_res["sparsity"] is not None:
                    sparsity_values.append(eval_res["sparsity"])
                    
                if eval_res["normalized_proximity"] is not None:
                    proximity_values.append(eval_res["normalized_proximity"])
                    
            if eval_res["generation_time_seconds"] is not None:
                generation_times.append(eval_res["generation_time_seconds"])
                
        validity_rate = (valid_counterfactuals / counterfactuals_found) if counterfactuals_found > 0 else 0.0
        feasibility_rate = (feasible_counterfactuals / counterfactuals_found) if counterfactuals_found > 0 else 0.0
        plausibility_rate = (plausible_counterfactuals / counterfactuals_found) if counterfactuals_found > 0 else 0.0
        
        avg_sparsity = float(np.mean(sparsity_values)) if sparsity_values else None
        avg_prox = float(np.mean(proximity_values)) if proximity_values else None
        avg_gen_time = float(np.mean(generation_times)) if generation_times else None
        
        return {
            "total_cases": total_cases,
            "anomalies_analyzed": anomalies_analyzed,
            "counterfactuals_found": counterfactuals_found,
            "valid_counterfactuals": valid_counterfactuals,
            "counterfactual_validity_rate": validity_rate,
            "feasible_counterfactuals": feasible_counterfactuals,
            "feasibility_rate": feasibility_rate,
            "plausible_counterfactuals": plausible_counterfactuals,
            "plausibility_rate": plausibility_rate,
            "average_sparsity": avg_sparsity,
            "average_normalized_proximity": avg_prox,
            "average_generation_time_seconds": avg_gen_time
        }
