"""
Analyze GPT accuracy from evaluation results.
Extracts GPT verdicts and compares against ground truth.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ROC AUC will not be calculated.")


class GPTAccuracyAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self):
        """Load ground truth from all summary.json files in runs directory."""
        ground_truth = {}

        # Define run configs (matching data_collector.py)
        run_configs = [
            {
                "run_name": "sora_pro_sora_pro_subset_run_3",
                "summary_path": "sora_pro_output_map_sora_pro_subset_run_3",
            },
            {
                "run_name": "veo_veo_run_6",
                "summary_path": "veo_output_map_veo_run_6",
            },
            {
                "run_name": "LingBot_lingbot_run_1",
                "summary_path": "LingBot_output_map_lingbot_run_1",
            },
            {
                "run_name": "HY-WorldPlay_hunyuan_run_2",
                "summary_path": "HY-WorldPlay_output_map_hunyuan_run_2",
            },
            {
                "run_name": "genie_genie_run_1",
                "summary_path": "genie",
            },
            {
                "run_name": "wan22_wan_run_1",
                "summary_path": "wan22_wan_run_1",
            }
        ]

        runs_dir = self.project_root / "runs"

        for run_config in run_configs:
            summary_path = runs_dir / run_config["summary_path"] / "summary.json"

            if not summary_path.exists():
                print(f"Warning: Summary not found at {summary_path}")
                continue

            with open(summary_path) as f:
                data = json.load(f)

            # Extract tasks from summary
            for task_info in data.get("tasks", []):
                task_id = task_info.get("task_id")
                # Only include tasks not ending with _00
                if not task_id.endswith("_00"):
                    key = (task_id, run_config["run_name"])
                    # Use human_state_evol as ground truth
                    ground_truth[key] = task_info.get("human_state_evol", False)

        print(f"Loaded ground truth for {len(ground_truth)} tasks")
        return ground_truth

    def analyze_gpt_accuracy(self, results_file: str):
        """Analyze GPT accuracy from results file."""
        with open(results_file) as f:
            data = json.load(f)

        results = data.get("results", [])

        # Collect GPT verdicts and ground truth
        gpt_verdicts = []
        ground_truth_labels = []
        gpt_scores = []  # For ROC AUC (if available)

        # Track by model
        by_model = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

        for result in results:
            task_id = result["task_id"]
            run_name = result["run_name"]
            key = (task_id, run_name)

            # Get ground truth
            if key not in self.ground_truth:
                print(f"Warning: No ground truth for {task_id} - {run_name}")
                continue

            ground_truth = self.ground_truth[key]

            # Extract GPT verdict
            gpt_verdict = None
            gpt_score = None

            # Check different result formats
            if "verification_results" in result:
                # Cross-model or double ensemble format
                verification = result["verification_results"]
                if "gpt_verdict" in verification:
                    gpt_verdict = verification["gpt_verdict"]
                elif "gpt_results" in verification:
                    # Double ensemble - extract GPT ensemble verdict
                    gpt_results = verification["gpt_results"]
                    if "verdicts" in gpt_results:
                        # Use majority vote from GPT ensemble
                        verdicts = [v for v in gpt_results["verdicts"] if v is not None]
                        if verdicts:
                            gpt_verdict = sum(verdicts) > len(verdicts) / 2
                            gpt_score = sum(verdicts) / len(verdicts)  # Vote fraction
            elif "verdict" in result:
                # Standalone GPT evaluator format
                gpt_verdict = result["verdict"]

            if gpt_verdict is None:
                print(f"Warning: No GPT verdict for {task_id} - {run_name}")
                continue

            # Collect for overall metrics
            gpt_verdicts.append(gpt_verdict)
            ground_truth_labels.append(ground_truth)
            if gpt_score is not None:
                gpt_scores.append(gpt_score)
            else:
                gpt_scores.append(1.0 if gpt_verdict else 0.0)

            # Update confusion matrix
            model_name = self._extract_model_name(run_name)
            if ground_truth and gpt_verdict:
                by_model[model_name]["tp"] += 1
            elif not ground_truth and gpt_verdict:
                by_model[model_name]["fp"] += 1
            elif not ground_truth and not gpt_verdict:
                by_model[model_name]["tn"] += 1
            elif ground_truth and not gpt_verdict:
                by_model[model_name]["fn"] += 1

        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(gpt_verdicts, ground_truth_labels, gpt_scores)

        # Calculate per-model metrics
        model_metrics = {}
        for model, confusion in by_model.items():
            tp, fp, tn, fn = confusion["tp"], confusion["fp"], confusion["tn"], confusion["fn"]
            total = tp + fp + tn + fn

            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            model_metrics[model] = {
                "confusion_matrix": confusion,
                "total": total,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        return {
            "overall": overall_metrics,
            "by_model": model_metrics,
            "num_tasks": len(gpt_verdicts)
        }

    def _calculate_metrics(self, verdicts, ground_truth, scores):
        """Calculate confusion matrix and metrics."""
        tp = sum(1 for v, gt in zip(verdicts, ground_truth) if v and gt)
        fp = sum(1 for v, gt in zip(verdicts, ground_truth) if v and not gt)
        tn = sum(1 for v, gt in zip(verdicts, ground_truth) if not v and not gt)
        fn = sum(1 for v, gt in zip(verdicts, ground_truth) if not v and gt)

        total = tp + fp + tn + fn

        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate ROC AUC if sklearn available
        roc_auc = None
        if SKLEARN_AVAILABLE and len(set(ground_truth)) > 1:
            try:
                roc_auc = roc_auc_score(ground_truth, scores)
            except:
                pass

        return {
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            },
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

    def _extract_model_name(self, run_name: str) -> str:
        """Extract simple model name from run_name."""
        run_lower = run_name.lower()
        if "veo" in run_lower:
            return "veo"
        elif "sora_pro" in run_lower:
            return "sora_pro"
        elif "sora" in run_lower:
            return "sora"
        elif "lingbot" in run_lower:
            return "lingbot"
        elif "hunyuan" in run_lower or "hy-" in run_lower:
            return "hunyuan"
        elif "genie" in run_lower:
            return "genie"
        elif "wan" in run_lower:
            return "wan22"
        else:
            return "unknown"

    def print_analysis(self, analysis: dict):
        """Print analysis results."""
        print("\n" + "=" * 60)
        print("GPT ACCURACY ANALYSIS")
        print("=" * 60)

        overall = analysis["overall"]
        cm = overall["confusion_matrix"]

        # Check if all ground truth is negative
        total_positive_gt = cm['tp'] + cm['fn']
        total_negative_gt = cm['tn'] + cm['fp']
        all_negative = total_positive_gt == 0

        print(f"\nGround Truth Distribution ({analysis['num_tasks']} tasks):")
        print(f"  Positive (pass=True):  {total_positive_gt}")
        print(f"  Negative (pass=False): {total_negative_gt}")

        if all_negative:
            print("\nNOTE: All ground truth labels are negative (fail).")
            print("      Precision/Recall/F1 are not meaningful in this case.")

        print(f"\nGPT Predictions:")
        gpt_positive = cm['tp'] + cm['fp']
        gpt_negative = cm['tn'] + cm['fn']
        print(f"  Predicted Pass: {gpt_positive} ({gpt_positive/analysis['num_tasks']*100:.1f}%)")
        print(f"  Predicted Fail: {gpt_negative} ({gpt_negative/analysis['num_tasks']*100:.1f}%)")

        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {overall['accuracy']:.1%}")

        if not all_negative:
            print(f"  Precision: {overall['precision']:.1%}")
            print(f"  Recall:    {overall['recall']:.1%}")
            print(f"  F1 Score:  {overall['f1']:.3f}")
        else:
            # Calculate false positive rate
            fpr = cm['fp'] / total_negative_gt if total_negative_gt > 0 else 0
            print(f"  False Positive Rate: {fpr:.1%} ({cm['fp']}/{total_negative_gt})")

        if overall.get("roc_auc") is not None:
            print(f"  ROC AUC:   {overall['roc_auc']:.3f}")

        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm['tn']} (correctly identified as fail)")
        print(f"  False Positives: {cm['fp']} (incorrectly said pass)")
        if not all_negative:
            print(f"  True Positives:  {cm['tp']} (correctly identified as pass)")
            print(f"  False Negatives: {cm['fn']} (incorrectly said fail)")

        print(f"\nPer-Model Performance:")
        by_model = analysis["by_model"]

        # Sort by accuracy
        sorted_models = sorted(by_model.items(), key=lambda x: x[1]["accuracy"], reverse=True)

        for model, metrics in sorted_models:
            mcm = metrics["confusion_matrix"]
            model_negative_gt = mcm['tn'] + mcm['fp']
            fpr = mcm['fp'] / model_negative_gt if model_negative_gt > 0 else 0

            print(f"\n  {model} ({metrics['total']} tasks):")
            print(f"    Accuracy:  {metrics['accuracy']:.1%}")
            print(f"    FP Rate:   {fpr:.1%} ({mcm['fp']}/{model_negative_gt})")
            print(f"    TN/FP:     {mcm['tn']}/{mcm['fp']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_gpt_accuracy.py <results_file>")
        sys.exit(1)

    results_file = sys.argv[1]
    project_root = "."

    analyzer = GPTAccuracyAnalyzer(project_root)
    analysis = analyzer.analyze_gpt_accuracy(results_file)
    analyzer.print_analysis(analysis)

    # Save to file
    output_file = results_file.replace(".json", "_gpt_accuracy.json")
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
