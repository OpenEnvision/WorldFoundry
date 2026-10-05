"""
Pairwise agreement analysis for model evaluation.
Compares pairwise model rankings based on ground truth vs Gemini verdicts.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class PairwiseAgreementAnalyzer:
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
                # Check for se_success first
                if "se_success" in task:
                    return task["se_success"]
                # Fall back to human_state_evol
                elif "human_state_evol" in task:
                    return task["human_state_evol"]
                else:
                    return None
        return None

    def organize_by_task(self, results_file: str) -> Dict[str, Dict[str, Dict]]:
        """
        Organize evaluation results by task_id.

        Returns:
            Dict mapping task_id -> run_name -> {gemini_verdict, ground_truth}
        """
        eval_results = self.load_evaluation_results(results_file)

        # Group by task_id and run_name
        task_data = defaultdict(dict)

        # First pass: collect all Gemini verdicts
        for result in eval_results["results"]:
            task_id = result["task_id"]
            run_name = result["run_name"]

            # Skip if verdict is missing
            if "verdict" not in result:
                print(f"Warning: Skipping {task_id} in {run_name} - missing verdict")
                continue

            gemini_verdict = result["verdict"]

            task_data[task_id][run_name] = {
                "gemini_verdict": gemini_verdict,
                "ground_truth": None  # Will fill in next
            }

        # Second pass: add ground truth labels
        summaries_cache = {}
        for task_id in task_data:
            for run_name in task_data[task_id]:
                # Load summary if not cached
                if run_name not in summaries_cache:
                    summaries_cache[run_name] = self.load_summary_json(run_name)

                summary = summaries_cache[run_name]
                ground_truth = self.get_ground_truth_label(summary, task_id)
                task_data[task_id][run_name]["ground_truth"] = ground_truth

        return dict(task_data)

    def compare_pairwise(self, score_a: bool, score_b: bool) -> str:
        """
        Compare two scores pairwise.

        Returns:
            "A_wins" if A > B
            "B_wins" if B > A
            "tie" if A == B
        """
        if score_a is None or score_b is None:
            return "unknown"

        if score_a == score_b:
            return "tie"
        elif score_a > score_b:  # True > False
            return "A_wins"
        else:
            return "B_wins"

    def calculate_agreement_score(self, gt_comparison: str, gemini_comparison: str) -> float:
        """
        Calculate agreement score between ground truth and Gemini comparisons.

        Returns:
            1.0 = full agreement (both say same thing)
            0.5 = partial agreement (one says tie, other differentiates)
            0.0 = reversal (opposite winners)
        """
        if gt_comparison == "unknown" or gemini_comparison == "unknown":
            return None

        # Full agreement
        if gt_comparison == gemini_comparison:
            return 1.0

        # Partial agreement: one is tie, the other is not
        if gt_comparison == "tie" or gemini_comparison == "tie":
            return 0.5

        # Reversal: complete disagreement (A_wins vs B_wins)
        return 0.0

    def get_model_pairs(self, run_names: List[str]) -> List[Tuple[str, str]]:
        """Generate all unique pairs of models."""
        pairs = []
        for i in range(len(run_names)):
            for j in range(i + 1, len(run_names)):
                pairs.append((run_names[i], run_names[j]))
        return pairs

    def simplify_run_name(self, run_name: str) -> str:
        """Simplify run name to just the model name."""
        if "sora_pro" in run_name:
            return "sora_pro"
        elif "veo" in run_name:
            return "veo"
        elif "LingBot" in run_name:
            return "lingbot"
        elif "HY-WorldPlay" in run_name or "hunyuan" in run_name:
            return "hunyuan"
        elif "genie" in run_name:
            return "genie"
        elif "wan" in run_name:
            return "wan22"
        return run_name

    def analyze_pairwise_agreement(self, results_file: str) -> Dict:
        """
        Analyze pairwise agreement between ground truth and Gemini rankings.

        Uses weighted scoring:
        - Full agreement: 1.0
        - Partial agreement (tie vs non-tie): 0.5
        - Reversal: 0.0

        Returns:
            Dictionary with pairwise comparison statistics
        """
        task_data = self.organize_by_task(results_file)

        # Get all unique run names
        all_run_names = set()
        for task_id in task_data:
            all_run_names.update(task_data[task_id].keys())
        all_run_names = sorted(all_run_names)

        # Generate all pairs
        model_pairs = self.get_model_pairs(all_run_names)

        # Track pairwise comparisons with weighted scores
        pairwise_stats = defaultdict(lambda: {
            "total_score": 0.0,
            "count": 0,
            "full_agreement": 0,
            "partial_agreement": 0,
            "reversal": 0,
            "unknown": 0,
            "details": []
        })

        overall_total_score = 0.0
        overall_count = 0
        overall_full = 0
        overall_partial = 0
        overall_reversal = 0
        overall_unknown = 0

        # For each task, compare all pairs
        for task_id, run_data in task_data.items():
            for model_a, model_b in model_pairs:
                # Skip if either model didn't run this task
                if model_a not in run_data or model_b not in run_data:
                    continue

                # Get scores
                data_a = run_data[model_a]
                data_b = run_data[model_b]

                # Compare based on ground truth
                gt_comparison = self.compare_pairwise(
                    data_a["ground_truth"],
                    data_b["ground_truth"]
                )

                # Compare based on Gemini verdict
                gemini_comparison = self.compare_pairwise(
                    data_a["gemini_verdict"],
                    data_b["gemini_verdict"]
                )

                # Calculate agreement score
                agreement_score = self.calculate_agreement_score(gt_comparison, gemini_comparison)

                pair_key = f"{self.simplify_run_name(model_a)}_vs_{self.simplify_run_name(model_b)}"

                if agreement_score is None:
                    pairwise_stats[pair_key]["unknown"] += 1
                    overall_unknown += 1
                else:
                    pairwise_stats[pair_key]["total_score"] += agreement_score
                    pairwise_stats[pair_key]["count"] += 1
                    overall_total_score += agreement_score
                    overall_count += 1

                    # Categorize
                    if agreement_score == 1.0:
                        pairwise_stats[pair_key]["full_agreement"] += 1
                        overall_full += 1
                    elif agreement_score == 0.5:
                        pairwise_stats[pair_key]["partial_agreement"] += 1
                        overall_partial += 1
                    else:  # 0.0
                        pairwise_stats[pair_key]["reversal"] += 1
                        overall_reversal += 1

                        # Track reversal details
                        pairwise_stats[pair_key]["details"].append({
                            "task_id": task_id,
                            "ground_truth": gt_comparison,
                            "gemini": gemini_comparison,
                            "score": agreement_score,
                            f"{self.simplify_run_name(model_a)}_gt": data_a["ground_truth"],
                            f"{self.simplify_run_name(model_b)}_gt": data_b["ground_truth"],
                            f"{self.simplify_run_name(model_a)}_gemini": data_a["gemini_verdict"],
                            f"{self.simplify_run_name(model_b)}_gemini": data_b["gemini_verdict"],
                        })

        # Calculate weighted agreement rates
        pairwise_agreement_rates = {}
        for pair, stats in pairwise_stats.items():
            if stats["count"] > 0:
                agreement_rate = stats["total_score"] / stats["count"]
            else:
                agreement_rate = 0
            pairwise_agreement_rates[pair] = agreement_rate

        overall_agreement_rate = overall_total_score / overall_count if overall_count > 0 else 0

        return {
            "overall": {
                "total_score": overall_total_score,
                "count": overall_count,
                "weighted_agreement_rate": overall_agreement_rate,
                "full_agreement": overall_full,
                "partial_agreement": overall_partial,
                "reversal": overall_reversal,
                "unknown": overall_unknown,
            },
            "by_pair": dict(pairwise_stats),
            "agreement_rates": pairwise_agreement_rates
        }

    def print_pairwise_analysis(self, analysis: Dict):
        """Pretty print the pairwise analysis."""
        print(f"\n{'='*70}")
        print("PAIRWISE AGREEMENT ANALYSIS (WEIGHTED)")
        print(f"{'='*70}")
        print("Scoring: Full agreement=1.0, Partial (tie vs non-tie)=0.5, Reversal=0.0")

        overall = analysis["overall"]
        print(f"\nOverall Statistics:")
        print(f"  Total pairwise comparisons: {overall['count']}")
        print(f"  Weighted agreement rate: {overall['weighted_agreement_rate']:.3f} ({overall['weighted_agreement_rate']*100:.1f}%)")
        print(f"  Full agreements: {overall['full_agreement']} (score=1.0)")
        print(f"  Partial agreements: {overall['partial_agreement']} (score=0.5)")
        print(f"  Reversals: {overall['reversal']} (score=0.0)")
        print(f"  Unknown: {overall['unknown']}")

        print(f"\n{'='*70}")
        print("Agreement by Model Pair:")
        print(f"{'='*70}")

        # Sort pairs by agreement rate
        sorted_pairs = sorted(
            analysis["agreement_rates"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for pair, agreement_rate in sorted_pairs:
            stats = analysis["by_pair"][pair]
            print(f"\n{pair}:")
            print(f"  Total comparisons: {stats['count']}")
            print(f"  Weighted agreement: {agreement_rate:.3f} ({agreement_rate*100:.1f}%)")
            print(f"  Full agreements: {stats['full_agreement']}")
            print(f"  Partial agreements: {stats['partial_agreement']}")
            print(f"  Reversals: {stats['reversal']}")

            # Show reversal examples
            if stats["details"]:
                print(f"  Reversals (wrong winner):")
                for detail in stats["details"][:3]:  # Show first 3
                    print(f"    - {detail['task_id']}: GT={detail['ground_truth']}, Gemini={detail['gemini']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pairwise_agreement.py <results_file.json>")
        print("Example: python pairwise_agreement.py evaluation_results_20260226_141217.json")
        sys.exit(1)

    results_file = sys.argv[1]
    project_root = "."

    analyzer = PairwiseAgreementAnalyzer(project_root)
    analysis = analyzer.analyze_pairwise_agreement(results_file)
    analyzer.print_pairwise_analysis(analysis)

    # Save analysis
    output_file = Path(results_file).stem + "_pairwise_agreement.json"
    output_path = Path(project_root) / "eval_se" / "results" / output_file

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
