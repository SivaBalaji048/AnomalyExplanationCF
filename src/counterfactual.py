"""Counterfactual generation module for Automated Anomaly Explanation.

Responsibilities:
- Generate constrained candidate counterfactuals.
- Optimize for validity, sparsity, and normalized proximity.
- Adhere strictly to configured bounds, step sizes, and mutability.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import yaml

# Resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES_CONFIG = PROJECT_ROOT / "config" / "features.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "config" / "model_config.yaml"


class CounterfactualEngine:
    """Deterministic counterfactual search engine."""

    def __init__(
        self,
        features_config_path: Optional[Path] = None,
        model_config_path: Optional[Path] = None,
    ):
        """Initialize the engine with configuration constraints.

        Parameters
        ----------
        features_config_path : Optional[Path]
            Path to features.yaml configuration file.
        model_config_path : Optional[Path]
            Path to model_config.yaml configuration file.
        """
        self.features_config_path = features_config_path or DEFAULT_FEATURES_CONFIG
        self.model_config_path = model_config_path or DEFAULT_MODEL_CONFIG

        with open(self.features_config_path, "r") as f:
            self.features_config = yaml.safe_load(f)

        with open(self.model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)

        self.feature_names = self.features_config["feature_sets"]["detector_inputs"]
        search_cfg = self.model_config["counterfactual_search"]
        self.max_candidates = search_cfg["max_candidates"]
        self.max_time = search_cfg["max_generation_time_seconds"]
        self.max_changed_features = search_cfg["max_changed_features"]
        self.ranking_priority = search_cfg["ranking_priority"]

        # Parse mutable features and physical bounds
        self.mutable_features = []
        self.feature_bounds = {}
        self.feature_steps = {}
        for fname, fconfig in self.features_config["features"].items():
            if fconfig.get("role") == "detector_input" and fconfig.get("mutable") is True:
                self.mutable_features.append(fname)
                self.feature_bounds[fname] = (fconfig["min"], fconfig["max"])
                self.feature_steps[fname] = fconfig["counterfactual_step"]

    def _generate_deltas(
        self, orig_val: float, min_val: float, max_val: float, step: float
    ) -> List[Tuple[float, int]]:
        """Generate valid perturbed values moving deterministically outwards.

        Parameters
        ----------
        orig_val : float
            Original physical value.
        min_val : float
            Minimum physical bound.
        max_val : float
            Maximum physical bound.
        step : float
            Configured step size.

        Returns
        -------
        List[Tuple[float, int]]
            Ordered candidate values within bounds and their step distance.
        """
        candidates = []
        i = 1

        while True:
            v_minus = orig_val - i * step
            v_plus = orig_val + i * step

            # Strictly enforce configured physical bounds
            if min_val <= v_minus <= max_val:
                candidates.append((v_minus, i))

            if min_val <= v_plus <= max_val:
                candidates.append((v_plus, i))

            # Break if we've permanently passed the bounds in both directions
            exhausted_minus = (v_minus < min_val)
            exhausted_plus = (v_plus > max_val)

            if exhausted_minus and exhausted_plus:
                break

            i += 1

        return candidates

    def _validate_original_state(self, original_state: pd.DataFrame) -> None:
        """Validate input state matches strict expected contract.

        Raises
        ------
        ValueError
            If dataframe structure, types, or completeness is invalid.
        """
        if not isinstance(original_state, pd.DataFrame):
            raise ValueError("Input state must be a pandas DataFrame.")
        if original_state.empty:
            raise ValueError("Input state DataFrame is empty.")
        if len(original_state) != 1:
            raise ValueError("Expected exactly one machine state (1 row).")
        if list(original_state.columns) != self.feature_names:
            raise ValueError(
                f"Columns must exactly match approved detector inputs in order.\n"
                f"Expected: {self.feature_names}\n"
                f"Received: {list(original_state.columns)}"
            )
        if original_state.isnull().values.any():
            raise ValueError("Input state contains missing values.")
        for col in original_state.columns:
            if not pd.api.types.is_numeric_dtype(original_state[col]):
                raise ValueError(f"Feature '{col}' must be numeric.")

    def generate(self, original_state: pd.DataFrame, preprocessor, detector, return_candidates: bool = False) -> Dict[str, Any]:
        """Search deterministically for the best valid counterfactual state.

        Parameters
        ----------
        original_state : pd.DataFrame
            Single-row DataFrame containing unscaled original measurements.
        preprocessor : DataPreprocessor
            Fitted data preprocessor to map physical to scaled units.
        detector : AnomalyDetector
            Fitted anomaly detector to determine validity.
        return_candidates : bool
            If True, includes the pool of all valid evaluated candidates in the result.

        Returns
        -------
        Dict[str, Any]
            Structured results containing status, counterfactual state, metrics,
            and explanations of changed features.
        """
        self._validate_original_state(original_state)

        if not getattr(preprocessor, "is_fitted", False):
            raise RuntimeError("Preprocessor must be fitted.")
        if not getattr(detector, "is_fitted", False):
            raise RuntimeError("Detector must be fitted.")

        start_time = time.time()
        candidates_evaluated = 0
        valid_candidates = []

        # 1. Check original state
        transformed_original = preprocessor.transform(original_state)
        is_anomaly = bool(detector.is_anomaly(transformed_original).iloc[0])
        original_score = float(detector.predict_anomaly_score(transformed_original).iloc[0])
        original_label = int(detector.predict_labels(transformed_original).iloc[0])

        if not is_anomaly:
            res = self._build_result(
                found=False,
                reason="original_state_is_already_normal",
                original=original_state,
                orig_score=original_score,
                orig_label=original_label,
                generation_time=time.time() - start_time,
                evaluated=0,
            )
            if return_candidates:
                res["candidates"] = []
            return res

        DEFAULT_BATCH_SIZE = 256

        # Helper to process a batch of candidates
        def _evaluate_candidate_batch(batch: List[Tuple[Dict[str, float], int]]) -> bool:
            """Evaluate a batch of candidates and return True if search budget allows continuing."""
            nonlocal candidates_evaluated, valid_candidates
            
            if not batch:
                return True
                
            if time.time() - start_time >= self.max_time:
                return False

            remaining_budget = self.max_candidates - candidates_evaluated
            if len(batch) > remaining_budget:
                batch = batch[:remaining_budget]

            batch_size = len(batch)
            if batch_size == 0:
                return False

            # Create batched DataFrame
            batch_df = pd.concat([original_state] * batch_size, ignore_index=True)
            for i, (changes_dict, sparsity) in enumerate(batch):
                for feat, new_val in changes_dict.items():
                    batch_df.at[i, feat] = new_val

            candidates_evaluated += batch_size

            # Candidate validity determined strictly by the trained detector in scaled space.
            # Intentional batch inference using the SAME fitted detector model.
            transformed_batch = preprocessor.transform(batch_df)
            is_cand_anomaly_series = detector.is_anomaly(transformed_batch)
            is_cand_anomaly = is_cand_anomaly_series.values
            
            if not is_cand_anomaly.all():
                # Identify indices of normal candidates
                normal_indices = [idx for idx, is_anom in enumerate(is_cand_anomaly) if not is_anom]
                
                # Predict anomaly scores ONLY for normal candidates.
                # If using detector.model.predict() directly for efficient batch inference, 
                # this is acceptable because it is the exact same fitted model.
                normal_transformed = transformed_batch.iloc[normal_indices]
                cand_scores = detector.predict_anomaly_score(normal_transformed).values
                
                # Their predicted label is already known to be detector.normal_label
                normal_label = detector.normal_label

                for idx_in_normal, i in enumerate(normal_indices):
                    cand_score = float(cand_scores[idx_in_normal])
                    cand_label = int(normal_label)
                    
                    changes_dict, sparsity = batch[i]
                    cand_row = batch_df.iloc[[i]].copy()
                    
                    # 1. PRESERVE ORIGINAL DATAFRAME INDEX
                    cand_row.index = original_state.index

                    # Calculate normalized proximity
                    norm_prox = 0.0
                    feature_details = []
                    
                    for k in self.mutable_features:
                        orig_k = float(original_state[k].iloc[0])
                        cand_k = float(cand_row[k].iloc[0])
                        
                        if orig_k != cand_k:
                            f_min, f_max = self.feature_bounds[k]
                            delta = cand_k - orig_k
                            range_span = f_max - f_min
                            n_delta = abs(delta) / range_span if range_span > 0 else 0.0
                            norm_prox += n_delta

                            feature_details.append({
                                "feature": k,
                                "original_value": orig_k,
                                "counterfactual_value": cand_k,
                                "delta": delta,
                                "normalized_delta": n_delta,
                            })

                    valid_candidates.append({
                        "state": cand_row,
                        "score": cand_score,
                        "label": cand_label,
                        "sparsity": sparsity,
                        "normalized_proximity": norm_prox,
                        "changed_features": feature_details,
                    })
            
            if candidates_evaluated >= self.max_candidates:
                return False
            if time.time() - start_time >= self.max_time:
                return False
                
            return True

        # Generate base deltas for all mutable features
        cands_by_feature = {}
        for feat in self.mutable_features:
            orig_val = float(original_state[feat].iloc[0])
            min_val, max_val = self.feature_bounds[feat]
            step = self.feature_steps[feat]
            cands_by_feature[feat] = self._generate_deltas(orig_val, min_val, max_val, step)

        # 2. Level 1 Search (Single feature modifications prioritized by step distance)
        level_1_queue = []
        for feat in self.mutable_features:
            for v, s in cands_by_feature[feat]:
                level_1_queue.append(({feat: v}, 1, s))
                
        level_1_queue.sort(key=lambda x: x[2])  # Prioritize smaller step changes across all features

        level_1_batch = []
        for changes_dict, sparsity, _ in level_1_queue:
            level_1_batch.append((changes_dict, sparsity))
            if len(level_1_batch) >= DEFAULT_BATCH_SIZE:
                if not _evaluate_candidate_batch(level_1_batch):
                    level_1_batch = []
                    break
                level_1_batch = []
                
        if level_1_batch:
            _evaluate_candidate_batch(level_1_batch)

        # 3. Level 2 Search (Lazy combinations, only if allowed and Level 1 yielded nothing)
        if not valid_candidates and self.max_changed_features >= 2 and len(self.mutable_features) >= 2:
            import itertools
            
            maps_by_feature = {}
            for feat in self.mutable_features:
                f_map = defaultdict(list)
                for v, s in cands_by_feature[feat]:
                    f_map[s].append(v)
                maps_by_feature[feat] = f_map
                
            max_possible_steps = 0
            pairs = list(itertools.combinations(self.mutable_features, 2))
            for f1, f2 in pairs:
                map1 = maps_by_feature[f1]
                map2 = maps_by_feature[f2]
                if map1 and map2:
                    max_possible_steps = max(max_possible_steps, max(map1.keys()) + max(map2.keys()))
                    
            lazy_search_aborted = False
            level_2_batch = []
            
            for total_steps in range(2, max_possible_steps + 1):
                if lazy_search_aborted:
                    break
                    
                for f1, f2 in pairs:
                    map1 = maps_by_feature[f1]
                    map2 = maps_by_feature[f2]
                    
                    if lazy_search_aborted:
                        break
                        
                    for steps1 in range(1, total_steps):
                        steps2 = total_steps - steps1
                        if steps1 in map1 and steps2 in map2:
                            for v1 in map1[steps1]:
                                for v2 in map2[steps2]:
                                    level_2_batch.append(({f1: v1, f2: v2}, 2))
                                    if len(level_2_batch) >= DEFAULT_BATCH_SIZE:
                                        if not _evaluate_candidate_batch(level_2_batch):
                                            lazy_search_aborted = True
                                            level_2_batch = []
                                            break
                                        level_2_batch = []
                                if lazy_search_aborted:
                                    break
                        if lazy_search_aborted:
                            break
                            
            if level_2_batch and not lazy_search_aborted:
                _evaluate_candidate_batch(level_2_batch)

        generation_time = time.time() - start_time

        if not valid_candidates:
            reason = "no_valid_counterfactual_found"
            if candidates_evaluated >= self.max_candidates or generation_time >= self.max_time:
                reason = "search_budget_exhausted"

            res = self._build_result(
                found=False,
                reason=reason,
                original=original_state,
                orig_score=original_score,
                orig_label=original_label,
                generation_time=generation_time,
                evaluated=candidates_evaluated,
            )
            if return_candidates:
                res["candidates"] = []
            return res

        # 5. Rank and return best candidate
        # Rank deterministically by available criteria (sparsity, then normalized_proximity)
        valid_candidates.sort(key=lambda x: (x["sparsity"], x["normalized_proximity"]))
        best = valid_candidates[0]

        res = self._build_result(
            found=True,
            reason="counterfactual_found",
            original=original_state,
            orig_score=original_score,
            orig_label=original_label,
            generation_time=generation_time,
            evaluated=candidates_evaluated,
            best_candidate=best,
        )
        if return_candidates:
            res["candidates"] = valid_candidates
        return res

    def _build_result(
        self,
        found: bool,
        reason: str,
        original: pd.DataFrame,
        orig_score: float,
        orig_label: int,
        generation_time: float,
        evaluated: int,
        best_candidate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct the standardized structured response dictionary."""
        res = {
            "found": found,
            "reason": reason,
            "original_state": original,
            "original_score": orig_score,
            "original_label": orig_label,
            "candidates_evaluated": evaluated,
            "generation_time_seconds": generation_time,
            "feasibility_status": "pending" if found else None,
            "plausibility_status": "pending" if found else None,
        }

        if best_candidate:
            res.update({
                "counterfactual_state": best_candidate["state"],
                "changed_features": best_candidate["changed_features"],
                "counterfactual_score": best_candidate["score"],
                "counterfactual_label": best_candidate["label"],
                "sparsity": best_candidate["sparsity"],
                "normalized_proximity": best_candidate["normalized_proximity"],
            })
        else:
            res.update({
                "counterfactual_state": None,
                "changed_features": [],
                "counterfactual_score": None,
                "counterfactual_label": None,
                "sparsity": None,
                "normalized_proximity": None,
            })

        return res
