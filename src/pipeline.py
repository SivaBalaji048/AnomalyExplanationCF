"""Pipeline module.

Responsibilities:
- connect the complete ML workflow into one end-to-end pipeline
"""
from typing import Dict, Any
import pandas as pd
from src.counterfactual import CounterfactualEngine
from src.feasibility import FeasibilityChecker
from src.plausibility import PlausibilityChecker

class ExplanationPipeline:
    """Orchestrates the end-to-end anomaly explanation process."""
    
    def __init__(
        self,
        preprocessor,
        detector,
        engine: CounterfactualEngine,
        feasibility_checker: FeasibilityChecker,
        plausibility_checker: PlausibilityChecker
    ):
        self.preprocessor = preprocessor
        self.detector = detector
        self.engine = engine
        self.feasibility_checker = feasibility_checker
        self.plausibility_checker = plausibility_checker

    def analyze(self, sample: pd.DataFrame) -> Dict[str, Any]:
        """Run the end-to-end analysis on a single observation.
        
        Parameters
        ----------
        sample : pd.DataFrame
            The physical observation to analyze.
            
        Returns
        -------
        Dict[str, Any]
            The explanation result, including the best counterfactual candidate if found.
        """
        # Generate candidates using the counterfactual engine
        result = self.engine.generate(
            original_state=sample,
            preprocessor=self.preprocessor,
            detector=self.detector,
            return_candidates=True
        )

        if not result["found"] or not result.get("candidates"):
            # Clean up the candidates list before returning if it was present
            result.pop("candidates", None)
            result["plausibility_distance"] = None
            result["plausibility_threshold"] = None
            result["plausibility_method"] = None
            return result
        
        original_state = result["original_state"]
        evaluated_candidates = []

        for cand in result["candidates"]:
            cand_state = cand["state"]
            
            # 1. Check Feasibility
            feas_result = self.feasibility_checker.check(original_state, cand_state)
            is_feasible = feas_result["feasible"]
            
            # 2. Check Plausibility
            plaus_result = self.plausibility_checker.check(cand_state)
            is_plausible = plaus_result["plausible"]
            plaus_dist = plaus_result["distance"]
            plaus_threshold = plaus_result["threshold"]
            plaus_method = plaus_result["method"]
            
            # Record results
            cand["is_feasible"] = is_feasible
            cand["is_plausible"] = is_plausible
            cand["plausibility_distance"] = plaus_dist
            cand["plausibility_threshold"] = plaus_threshold
            cand["plausibility_method"] = plaus_method
            
            # Ranking key based on project specification:
            # 1. detector validity (all candidates are already valid by generation)
            # 2. feasibility (prefer feasible -> False is sorted before True in Python, so use `not is_feasible`)
            # 3. plausibility (prefer plausible -> `not is_plausible`)
            # 4. sparsity (prefer lower)
            # 5. normalized proximity (prefer lower)
            sort_key = (
                not is_feasible,
                not is_plausible,
                cand["sparsity"],
                cand["normalized_proximity"]
            )
            
            evaluated_candidates.append((sort_key, cand))
            
        # Rank candidates deterministically
        evaluated_candidates.sort(key=lambda x: x[0])
        best_cand = evaluated_candidates[0][1]
        
        # Update the main result dictionary with the best candidate's details
        result["counterfactual_state"] = best_cand["state"]
        result["changed_features"] = best_cand["changed_features"]
        result["counterfactual_score"] = best_cand["score"]
        result["counterfactual_label"] = best_cand["label"]
        result["sparsity"] = best_cand["sparsity"]
        result["normalized_proximity"] = best_cand["normalized_proximity"]
        
        # Distinguish feasibility and plausibility statuses
        result["feasibility_status"] = "feasible" if best_cand["is_feasible"] else "infeasible"
        result["plausibility_status"] = "plausible" if best_cand["is_plausible"] else "implausible"
        result["plausibility_distance"] = best_cand["plausibility_distance"]
        result["plausibility_threshold"] = best_cand["plausibility_threshold"]
        result["plausibility_method"] = best_cand["plausibility_method"]
        
        # Clean up the candidates list before returning
        result.pop("candidates", None)
        
        return result
