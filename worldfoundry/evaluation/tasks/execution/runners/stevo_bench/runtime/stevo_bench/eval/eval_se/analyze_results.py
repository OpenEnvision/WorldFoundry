"""
Analyze evaluation results and compare with ground truth labels.
Creates confusion matrix comparing Gemini verdicts vs. ground truth.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
try:
    from sklearn.metrics import roc_auc_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. ROC AUC calculation will be skipped.")


class ResultsAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.runs_dir = self.project_root / "runs"

    def load_evaluation_results(self, results_file: str) -> Dict:
        """Load the evaluation results JSON file."""
        results_path = Path(results_file)
        if not results_path.is_absolute():
            results_path = self.project_root / "eval_se" / "results" / results_file

        with open(results_path, 'r') as f:
            return json.load(f)

    def load_summary_json(self, run_name: str) -> Dict:
        """Load summary.json for a specific run."""
        # Map run names to summary paths
        run_mapping = {
            "sora_pro_sora_pro_subset_run_3": "sora_pro_output_map_sora_pro_subset_run_3",
            "veo_veo_run_6": "veo_output_map_veo_run_6",
            "LingBot_lingbot_run_1": "LingBot_output_map_lingbot_run_1",
            "HY-WorldPlay_hunyuan_run_2": "HY-WorldPlay_output_map_hunyuan_run_2",
            "genie_genie_run_1": "genie",
            "wan22_wan_run_1": "wan22_wan_run_1"
        }

        summary_path_key = run_mapping.get(run_name, run_name)
        summary_path = self.runs_dir / summary_path_key / "summary.json"

        if not summary_path.exists():
            print(f"Warning: Could not find summary.json at {summary_path}")
            return {}

        with open(summary_path, 'r') as f:
            return json.load(f)

    def get_ground_truth_label(self, summary: Dict, task_id: str) -> bool:
        """
        Get ground truth label for a task from summary.json.
        Checks for se_success field, falls back to human_state_evol.
        """
        for task in summary.get("tasks", []):
            if task.get("task_id") == task_id:
                # Check for se_success first
                if "se_success" in task:
                    return task["se_success"]
                # Fall back to human_state_evol
                elif "human_state_evol" in task:
                    return task["human_state_evol"]
                else:
                    return None

        return None

    def create_confusion_matrix(self, results_file: str) -> Dict:
        """
        Create confusion matrix comparing Gemini verdicts vs ground truth.
        Also calculates ROC AUC if scores are available.

        Returns:
            Dictionary with confusion matrix and metrics
        """
        # Load evaluation results
        eval_results = self.load_evaluation_results(results_file)

        # Check if ensemble results (has scores)
        is_ensemble = False
        is_cross_model = False
        if eval_results["results"]:
            first_result = eval_results["results"][0]
            if "ensemble_results" in first_result:
                is_ensemble = True
            elif "verification_results" in first_result:
                # Check if it's cross-model (has both gemini_verdict and gpt_verdict)
                if "gemini_verdict" in first_result["verification_results"] and "gpt_verdict" in first_result["verification_results"]:
                    is_cross_model = True

        # Initialize confusion matrix
        # [predicted_negative, predicted_positive] x [actual_negative, actual_positive]
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        # Track details for each prediction
        detailed_results = []
        ground_truth_field_used = None

        # For ROC AUC calculation
        y_true = []
        y_scores = []

        # Group results by run
        results_by_run = {}
        for result in eval_results["results"]:
            run_name = result["run_name"]
            if run_name not in results_by_run:
                results_by_run[run_name] = []
            results_by_run[run_name].append(result)

        # Compare each result with ground truth
        for run_name, run_results in results_by_run.items():
            summary = self.load_summary_json(run_name)

            for result in run_results:
                task_id = result["task_id"]

                # Check if verdict exists
                if "verdict" not in result:
                    print(f"Warning: No verdict for {task_id} in {run_name}")
                    continue

                gemini_verdict = result["verdict"]

                # Get ground truth
                ground_truth = self.get_ground_truth_label(summary, task_id)

                if ground_truth is None:
                    print(f"Warning: No ground truth for {task_id} in {run_name}")
                    continue

                # Determine which field was used (for reporting)
                if ground_truth_field_used is None:
                    for task in summary.get("tasks", []):
                        if task.get("task_id") == task_id:
                            if "se_success" in task:
                                ground_truth_field_used = "se_success"
                            else:
                                ground_truth_field_used = "human_state_evol"
                            break

                # Calculate score for ROC AUC (if ensemble or cross-model)
                score = None
                if is_ensemble and "ensemble_results" in result:
                    verdicts = result["ensemble_results"].get("verdicts", [])
                    valid_verdicts = [v for v in verdicts if v is not None]
                    if valid_verdicts:
                        # Use vote fraction as score (0.0, 0.333, 0.667, 1.0)
                        score = sum(1 for v in valid_verdicts if v) / len(valid_verdicts)
                elif is_cross_model and "verification_results" in result:
                    # Cross-model: both yes=1.0, one yes/one no=0.5, both no=0.0
                    verification = result["verification_results"]
                    gemini_v = verification.get("gemini_verdict", False)
                    gpt_v = verification.get("gpt_verdict", False)

                    if gemini_v and gpt_v:
                        score = 1.0  # Both yes
                    elif gemini_v or gpt_v:
                        score = 0.5  # One yes, one no
                    else:
                        score = 0.0  # Both no

                # Collect for ROC AUC
                if score is not None:
                    y_true.append(int(ground_truth))
                    y_scores.append(score)

                # Update confusion matrix
                if gemini_verdict and ground_truth:
                    true_positives += 1
                    outcome = "TP"
                elif gemini_verdict and not ground_truth:
                    false_positives += 1
                    outcome = "FP"
                elif not gemini_verdict and not ground_truth:
                    true_negatives += 1
                    outcome = "TN"
                else:  # not gemini_verdict and ground_truth
                    false_negatives += 1
                    outcome = "FN"

                detailed_results.append({
                    "task_id": task_id,
                    "run_name": run_name,
                    "gemini_verdict": gemini_verdict,
                    "ground_truth": ground_truth,
                    "outcome": outcome,
                    "score": score
                })

        # Calculate metrics
        total = true_positives + false_positives + true_negatives + false_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate ROC AUC if scores available
        roc_auc = None
        roc_curve_data = None
        if SKLEARN_AVAILABLE and len(y_true) > 0 and len(set(y_true)) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_curve_data = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist()
                }
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")

        result = {
            "ground_truth_field": ground_truth_field_used,
            "is_ensemble": is_ensemble,
            "is_cross_model": is_cross_model,
            "confusion_matrix": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives
            },
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "total_evaluated": total,
            "detailed_results": detailed_results
        }

        if roc_auc is not None:
            result["metrics"]["roc_auc"] = roc_auc
            result["roc_curve"] = roc_curve_data

        return result

    def print_confusion_matrix(self, analysis: Dict):
        """Pretty print the confusion matrix and metrics."""
        cm = analysis["confusion_matrix"]
        metrics = analysis["metrics"]

        print(f"\n{'='*60}")
        print("CONFUSION MATRIX ANALYSIS")
        print(f"{'='*60}")
        print(f"Ground truth field used: {analysis['ground_truth_field']}")
        print(f"Total evaluated: {analysis['total_evaluated']}")
        if analysis.get("is_cross_model"):
            print("Evaluation type: CROSS-MODEL ENSEMBLE (Gemini + GPT)")
            print("  ROC AUC scoring: both yes=1.0, one yes/one no=0.5, both no=0.0")
        elif analysis.get("is_ensemble"):
            print("Evaluation type: ENSEMBLE (with confidence scores)")
        else:
            print("Evaluation type: STANDARD (binary only)")
        print()

        # Print confusion matrix
        print("Confusion Matrix:")
        print("                    Predicted")
        print("                 Negative  Positive")
        print(f"Actual Negative  {cm['true_negatives']:^8}  {cm['false_positives']:^8}")
        print(f"Actual Positive  {cm['false_negatives']:^8}  {cm['true_positives']:^8}")
        print()

        # Print metrics
        print("Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1_score']:.3f}")
        if "roc_auc" in metrics:
            print(f"  ROC AUC:   {metrics['roc_auc']:.3f}")
        elif analysis.get("is_ensemble") or analysis.get("is_cross_model"):
            print(f"  ROC AUC:   Not available (sklearn not installed)")
        else:
            print(f"  ROC AUC:   Not applicable (binary predictions only)")
        print()

        # Print breakdown by outcome
        outcomes = {}
        for result in analysis["detailed_results"]:
            outcome = result["outcome"]
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        print("Breakdown:")
        if "TP" in outcomes:
            print(f"  True Positives:  {outcomes['TP']}")
        if "FP" in outcomes:
            print(f"  False Positives: {outcomes['FP']}")
        if "TN" in outcomes:
            print(f"  True Negatives:  {outcomes['TN']}")
        if "FN" in outcomes:
            print(f"  False Negatives: {outcomes['FN']}")

        # Show score distribution for cross-model
        if analysis.get("is_cross_model"):
            scores = [r.get("score") for r in analysis["detailed_results"] if r.get("score") is not None]
            if scores:
                score_counts = {
                    1.0: sum(1 for s in scores if s == 1.0),
                    0.5: sum(1 for s in scores if s == 0.5),
                    0.0: sum(1 for s in scores if s == 0.0)
                }
                print("\nModel Agreement Distribution:")
                print(f"  Both agree (Yes):  {score_counts[1.0]} (score=1.0)")
                print(f"  Disagree:          {score_counts[0.5]} (score=0.5)")
                print(f"  Both agree (No):   {score_counts[0.0]} (score=0.0)")

        # Show false positives and false negatives
        print()
        fps = [r for r in analysis["detailed_results"] if r["outcome"] == "FP"]
        fns = [r for r in analysis["detailed_results"] if r["outcome"] == "FN"]

        if fps:
            print(f"\nFalse Positives ({len(fps)}):")
            for r in fps:
                print(f"  - {r['task_id']} ({r['run_name']})")

        if fns:
            print(f"\nFalse Negatives ({len(fns)}):")
            for r in fns:
                print(f"  - {r['task_id']} ({r['run_name']})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.json>")
        print("Example: python analyze_results.py evaluation_results_20240226_123456.json")
        sys.exit(1)

    results_file = sys.argv[1]
    project_root = "."

    analyzer = ResultsAnalyzer(project_root)
    analysis = analyzer.create_confusion_matrix(results_file)
    analyzer.print_confusion_matrix(analysis)

    # Save analysis
    output_file = Path(results_file).stem + "_analysis.json"
    output_path = Path(project_root) / "eval_se" / "results" / output_file

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()
