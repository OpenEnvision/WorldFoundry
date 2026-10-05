"""
Analyze individual judge performance in ensemble evaluation.
Calculate confusion matrix for each of the 3 judges separately.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class EnsembleJudgeAnalyzer:
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
        """Get ground truth label for a task from summary.json."""
        for task in summary.get("tasks", []):
            if task.get("task_id") == task_id:
                if "se_success" in task:
                    return task["se_success"]
                elif "human_state_evol" in task:
                    return task["human_state_evol"]
                else:
                    return None
        return None

    def analyze_judges(self, results_file: str) -> Dict:
        """
        Analyze performance of each judge separately.

        Returns:
            Dictionary with confusion matrices for each judge and ensemble
        """
        eval_results = self.load_evaluation_results(results_file)

        # Check if this is ensemble results
        if not eval_results["results"] or "ensemble_results" not in eval_results["results"][0]:
            print("ERROR: This is not an ensemble evaluation results file!")
            return None

        # Initialize confusion matrices for each judge + ensemble
        judges_stats = {
            "judge_1": {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "details": []},
            "judge_2": {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "details": []},
            "judge_3": {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "details": []},
            "ensemble": {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "details": []},
        }

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

                # Get ground truth
                ground_truth = self.get_ground_truth_label(summary, task_id)

                if ground_truth is None:
                    print(f"Warning: No ground truth for {task_id} in {run_name}")
                    continue

                # Get ensemble results
                ensemble_results = result.get("ensemble_results", {})
                verdicts = ensemble_results.get("verdicts", [])

                if len(verdicts) < 3:
                    print(f"Warning: Not enough verdicts for {task_id} in {run_name}")
                    continue

                # Get ensemble final verdict
                ensemble_verdict = result.get("verdict")

                # Update stats for each judge
                for i, judge_verdict in enumerate(verdicts[:3], 1):
                    judge_key = f"judge_{i}"

                    if judge_verdict is None:
                        continue

                    # Update confusion matrix
                    if judge_verdict and ground_truth:
                        judges_stats[judge_key]["TP"] += 1
                        outcome = "TP"
                    elif judge_verdict and not ground_truth:
                        judges_stats[judge_key]["FP"] += 1
                        outcome = "FP"
                    elif not judge_verdict and not ground_truth:
                        judges_stats[judge_key]["TN"] += 1
                        outcome = "TN"
                    else:  # not judge_verdict and ground_truth
                        judges_stats[judge_key]["FN"] += 1
                        outcome = "FN"

                    judges_stats[judge_key]["details"].append({
                        "task_id": task_id,
                        "run_name": run_name,
                        "verdict": judge_verdict,
                        "ground_truth": ground_truth,
                        "outcome": outcome
                    })

                # Update stats for ensemble
                if ensemble_verdict is not None:
                    if ensemble_verdict and ground_truth:
                        judges_stats["ensemble"]["TP"] += 1
                        outcome = "TP"
                    elif ensemble_verdict and not ground_truth:
                        judges_stats["ensemble"]["FP"] += 1
                        outcome = "FP"
                    elif not ensemble_verdict and not ground_truth:
                        judges_stats["ensemble"]["TN"] += 1
                        outcome = "TN"
                    else:
                        judges_stats["ensemble"]["FN"] += 1
                        outcome = "FN"

                    judges_stats["ensemble"]["details"].append({
                        "task_id": task_id,
                        "run_name": run_name,
                        "verdict": ensemble_verdict,
                        "ground_truth": ground_truth,
                        "outcome": outcome
                    })

        # Calculate metrics for each judge
        results = {}
        for judge_name, stats in judges_stats.items():
            tp = stats["TP"]
            fp = stats["FP"]
            tn = stats["TN"]
            fn = stats["FN"]
            total = tp + fp + tn + fn

            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results[judge_name] = {
                "confusion_matrix": {
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn
                },
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                },
                "total": total,
                "details": stats["details"]
            }

        return results

    def print_judge_comparison(self, analysis: Dict):
        """Pretty print comparison of judges."""
        print(f"\n{'='*70}")
        print("ENSEMBLE JUDGE COMPARISON")
        print(f"{'='*70}")

        # Print table header
        print(f"\n{'Judge':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 70)

        # Print each judge
        judge_names = {
            "judge_1": "Judge 1 (Conservative)",
            "judge_2": "Judge 2 (Evidence)",
            "judge_3": "Judge 3 (Analytical)",
            "ensemble": "Ensemble (Majority)"
        }

        for judge_key in ["judge_1", "judge_2", "judge_3", "ensemble"]:
            if judge_key not in analysis:
                continue

            judge_data = analysis[judge_key]
            metrics = judge_data["metrics"]

            display_name = judge_names.get(judge_key, judge_key)
            print(f"{display_name:<15} "
                  f"{metrics['accuracy']:<12.3f} "
                  f"{metrics['precision']:<12.3f} "
                  f"{metrics['recall']:<12.3f} "
                  f"{metrics['f1_score']:<12.3f}")

        # Detailed confusion matrices
        print(f"\n{'='*70}")
        print("DETAILED CONFUSION MATRICES")
        print(f"{'='*70}")

        for judge_key in ["judge_1", "judge_2", "judge_3", "ensemble"]:
            if judge_key not in analysis:
                continue

            judge_data = analysis[judge_key]
            cm = judge_data["confusion_matrix"]
            metrics = judge_data["metrics"]

            print(f"\n{judge_names.get(judge_key, judge_key)}:")
            print(f"  Total evaluated: {judge_data['total']}")
            print(f"  Confusion Matrix:")
            print(f"                     Predicted")
            print(f"                  Negative  Positive")
            print(f"  Actual Negative  {cm['TN']:^8}  {cm['FP']:^8}")
            print(f"  Actual Positive  {cm['FN']:^8}  {cm['TP']:^8}")
            print(f"  Metrics:")
            print(f"    Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall:    {metrics['recall']:.3f}")
            print(f"    F1 Score:  {metrics['f1_score']:.3f}")

        # Find disagreements between judges
        print(f"\n{'='*70}")
        print("JUDGE DISAGREEMENTS")
        print(f"{'='*70}")

        # Compare each judge with ensemble
        for judge_key in ["judge_1", "judge_2", "judge_3"]:
            judge_details = analysis[judge_key]["details"]
            ensemble_details = analysis["ensemble"]["details"]

            # Create lookup for ensemble verdicts
            ensemble_lookup = {
                (d["task_id"], d["run_name"]): d["verdict"]
                for d in ensemble_details
            }

            disagreements = []
            for detail in judge_details:
                task_key = (detail["task_id"], detail["run_name"])
                ensemble_verdict = ensemble_lookup.get(task_key)

                if ensemble_verdict is not None and detail["verdict"] != ensemble_verdict:
                    disagreements.append({
                        "task_id": detail["task_id"],
                        "run_name": detail["run_name"],
                        "judge_verdict": detail["verdict"],
                        "ensemble_verdict": ensemble_verdict,
                        "ground_truth": detail["ground_truth"]
                    })

            print(f"\n{judge_names[judge_key]} vs Ensemble: {len(disagreements)} disagreements")
            if disagreements:
                for d in disagreements[:5]:  # Show first 5
                    print(f"  {d['task_id']} ({d['run_name']}): "
                          f"Judge={'PASS' if d['judge_verdict'] else 'FAIL'}, "
                          f"Ensemble={'PASS' if d['ensemble_verdict'] else 'FAIL'}, "
                          f"GT={'PASS' if d['ground_truth'] else 'FAIL'}")
                if len(disagreements) > 5:
                    print(f"  ... and {len(disagreements) - 5} more")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ensemble_judges.py <results_file.json>")
        print("Example: python analyze_ensemble_judges.py evaluation_results_20260226_161252.json")
        sys.exit(1)

    results_file = sys.argv[1]
    project_root = "."

    analyzer = EnsembleJudgeAnalyzer(project_root)
    analysis = analyzer.analyze_judges(results_file)

    if analysis is None:
        sys.exit(1)

    analyzer.print_judge_comparison(analysis)

    # Save analysis
    output_file = Path(results_file).stem + "_judges_analysis.json"
    output_path = Path(project_root) / "eval_se" / "results" / output_file

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
