"""
Calculate model win rates based on Gemini and human scores.
Plots correlation between Gemini vs human assessments of model quality.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/scipy not available. Plotting will be skipped.")


class ModelWinRateAnalyzer:
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
        return None

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
                continue

            gemini_verdict = result["verdict"]

            task_data[task_id][run_name] = {
                "gemini_verdict": gemini_verdict,
                "ground_truth": None
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

    def calculate_win_rates(self, results_file: str) -> Dict:
        """
        Calculate win rates for each model against all others.

        Returns:
            Dict with win rates based on Gemini and human scores
        """
        task_data = self.organize_by_task(results_file)

        # Get all unique models
        all_models = set()
        for task_id in task_data:
            all_models.update(task_data[task_id].keys())
        all_models = sorted(all_models)

        # Simplify model names
        model_simple_names = {model: self.simplify_run_name(model) for model in all_models}

        # Initialize win counters
        model_stats = {}
        for model in all_models:
            simple_name = model_simple_names[model]
            model_stats[simple_name] = {
                "gemini_wins": 0,
                "gemini_losses": 0,
                "gemini_ties": 0,
                "human_wins": 0,
                "human_losses": 0,
                "human_ties": 0,
                "total_comparisons": 0
            }

        # For each task, compare all pairs of models
        for task_id, models_data in task_data.items():
            # Get all models that ran this task
            models_in_task = list(models_data.keys())

            # Compare each pair
            for i in range(len(models_in_task)):
                for j in range(i + 1, len(models_in_task)):
                    model_a = models_in_task[i]
                    model_b = models_in_task[j]

                    data_a = models_data[model_a]
                    data_b = models_data[model_b]

                    # Skip if missing data
                    if data_a["ground_truth"] is None or data_b["ground_truth"] is None:
                        continue

                    simple_a = model_simple_names[model_a]
                    simple_b = model_simple_names[model_b]

                    # Compare Gemini verdicts
                    gemini_a = data_a["gemini_verdict"]
                    gemini_b = data_b["gemini_verdict"]

                    if gemini_a and not gemini_b:
                        # A wins according to Gemini
                        model_stats[simple_a]["gemini_wins"] += 1
                        model_stats[simple_b]["gemini_losses"] += 1
                    elif gemini_b and not gemini_a:
                        # B wins according to Gemini
                        model_stats[simple_b]["gemini_wins"] += 1
                        model_stats[simple_a]["gemini_losses"] += 1
                    else:
                        # Tie
                        model_stats[simple_a]["gemini_ties"] += 1
                        model_stats[simple_b]["gemini_ties"] += 1

                    # Compare ground truth
                    human_a = data_a["ground_truth"]
                    human_b = data_b["ground_truth"]

                    if human_a and not human_b:
                        # A wins according to human
                        model_stats[simple_a]["human_wins"] += 1
                        model_stats[simple_b]["human_losses"] += 1
                    elif human_b and not human_a:
                        # B wins according to human
                        model_stats[simple_b]["human_wins"] += 1
                        model_stats[simple_a]["human_losses"] += 1
                    else:
                        # Tie
                        model_stats[simple_a]["human_ties"] += 1
                        model_stats[simple_b]["human_ties"] += 1

                    # Increment total comparisons
                    model_stats[simple_a]["total_comparisons"] += 1
                    model_stats[simple_b]["total_comparisons"] += 1

        # Calculate win rates
        model_win_rates = {}
        for model, stats in model_stats.items():
            total = stats["total_comparisons"]
            if total > 0:
                gemini_win_rate = stats["gemini_wins"] / total
                human_win_rate = stats["human_wins"] / total
            else:
                gemini_win_rate = 0
                human_win_rate = 0

            model_win_rates[model] = {
                "gemini_win_rate": gemini_win_rate,
                "human_win_rate": human_win_rate,
                "stats": stats
            }

        return model_win_rates

    def plot_and_analyze(self, model_win_rates: Dict, output_path: Path):
        """
        Plot model win rates and calculate correlation.
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping plot.")
            return None

        # Extract data
        models = list(model_win_rates.keys())
        human_scores = [model_win_rates[m]["human_win_rate"] for m in models]
        gemini_scores = [model_win_rates[m]["gemini_win_rate"] for m in models]

        # Calculate correlations
        pearson_r, pearson_p = pearsonr(human_scores, gemini_scores)
        spearman_rho, spearman_p = spearmanr(human_scores, gemini_scores)

        # Linear regression
        X = np.array(human_scores).reshape(-1, 1)
        y = np.array(gemini_scores)
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(human_scores, gemini_scores, s=200, alpha=0.6, c='blue')

        # Add model labels
        for i, model in enumerate(models):
            plt.annotate(model, (human_scores[i], gemini_scores[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        # Plot regression line
        plt.plot(human_scores, y_pred, 'r--', linewidth=2, label='Linear Fit')

        # Labels and title
        plt.xlabel('Human Win Rate (Ground Truth)', fontsize=12, fontweight='bold')
        plt.ylabel('Gemini Win Rate (Predicted)', fontsize=12, fontweight='bold')
        plt.title('Model Win Rates: Gemini vs Human Assessment', fontsize=14, fontweight='bold')

        # Add correlation info
        textstr = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})\n'
        textstr += f'Spearman ρ = {spearman_rho:.3f} (p={spearman_p:.4f})\n'
        textstr += f'R² = {reg.score(X, y):.3f}\n'
        textstr += f'Slope = {reg.coef_[0]:.3f}'
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Equal aspect for better comparison
        plt.axis('equal')
        plt.xlim(-0.05, max(max(human_scores), max(gemini_scores)) + 0.05)
        plt.ylim(-0.05, max(max(human_scores), max(gemini_scores)) + 0.05)

        # Add diagonal reference line (perfect agreement)
        max_val = max(max(human_scores), max(gemini_scores))
        plt.plot([0, max_val], [0, max_val], 'k:', linewidth=1, alpha=0.3, label='Perfect Agreement')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

        return {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "r_squared": reg.score(X, y),
            "slope": reg.coef_[0],
            "intercept": reg.intercept_
        }

    def print_win_rates(self, model_win_rates: Dict):
        """Pretty print win rates."""
        print(f"\n{'='*70}")
        print("MODEL WIN RATES")
        print(f"{'='*70}")

        print(f"\n{'Model':<15} {'Human Win %':<15} {'Gemini Win %':<15} {'Comparisons':<15}")
        print("-" * 70)

        # Sort by human win rate
        sorted_models = sorted(model_win_rates.items(),
                              key=lambda x: x[1]["human_win_rate"],
                              reverse=True)

        for model, data in sorted_models:
            human_wr = data["human_win_rate"] * 100
            gemini_wr = data["gemini_win_rate"] * 100
            total = data["stats"]["total_comparisons"]

            print(f"{model:<15} {human_wr:<15.1f} {gemini_wr:<15.1f} {total:<15}")

        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python model_win_rates.py <results_file.json>")
        print("Example: python model_win_rates.py evaluation_results_20260226_161252.json")
        sys.exit(1)

    results_file = sys.argv[1]
    project_root = "."

    analyzer = ModelWinRateAnalyzer(project_root)

    # Calculate win rates
    print("Calculating model win rates...")
    model_win_rates = analyzer.calculate_win_rates(results_file)

    # Print results
    analyzer.print_win_rates(model_win_rates)

    # Plot and analyze correlation
    output_path = Path(project_root) / "eval_se" / "results" / "model_win_rates_plot.png"
    correlation_stats = analyzer.plot_and_analyze(model_win_rates, output_path)

    if correlation_stats:
        print(f"\n{'='*70}")
        print("CORRELATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Pearson correlation (r):  {correlation_stats['pearson_r']:.3f} (p={correlation_stats['pearson_p']:.4f})")
        print(f"Spearman correlation (ρ): {correlation_stats['spearman_rho']:.3f} (p={correlation_stats['spearman_p']:.4f})")
        print(f"R-squared:                {correlation_stats['r_squared']:.3f}")
        print(f"Regression slope:         {correlation_stats['slope']:.3f}")
        print(f"Regression intercept:     {correlation_stats['intercept']:.3f}")

    # Save data
    output_file = Path(results_file).stem + "_win_rates.json"
    output_path_json = Path(project_root) / "eval_se" / "results" / output_file

    save_data = {
        "model_win_rates": model_win_rates,
        "correlation_stats": correlation_stats
    }

    with open(output_path_json, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nWin rates saved to: {output_path_json}")


if __name__ == "__main__":
    main()
