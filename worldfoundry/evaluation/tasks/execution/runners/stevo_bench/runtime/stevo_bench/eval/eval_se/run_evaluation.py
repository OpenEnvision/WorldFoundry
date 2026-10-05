"""
Main evaluation script for Gemini-based state evolution evaluation.
Orchestrates the two-step evaluation process and saves results.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from data_collector import TaskDataCollector
from gemini_evaluator import GeminiEvaluator
from gemini_evaluator_ensemble import GeminiEnsembleEvaluator
from gemini_evaluator_evidence import GeminiEvidenceEvaluator
from gemini_gpt_evaluator import GeminiGPTEvaluator
from gpt_evaluator import GPTEvaluator
from gemini_gpt_double_ensemble import GeminiGPTDoubleEnsemble


class EvaluationRunner:
    def __init__(self, project_root: str, output_dir: str = "eval_se/results",
                 use_ensemble: bool = False, use_evidence: bool = False, use_cross_model: bool = False,
                 use_gpt: bool = False, use_double_ensemble: bool = False, task_dir: str = "benchmark/tasks_v4"):
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_ensemble = use_ensemble
        self.use_evidence = use_evidence
        self.use_cross_model = use_cross_model
        self.use_gpt = use_gpt
        self.use_double_ensemble = use_double_ensemble

        self.collector = TaskDataCollector(project_root, task_dir)

        # Check for conflicting flags
        flags_set = sum([use_ensemble, use_evidence, use_cross_model, use_gpt, use_double_ensemble])
        if flags_set > 1:
            raise ValueError("Cannot use multiple evaluator flags together")

        if use_double_ensemble:
            self.evaluator = GeminiGPTDoubleEnsemble()
            print("Using DOUBLE ENSEMBLE (Gemini 3-prompts + GPT 3-prompts, unanimous cross-model)")
        elif use_gpt:
            self.evaluator = GPTEvaluator()
            print("Using GPT evaluator (GPT-5.2 with frame extraction)")
        elif use_cross_model:
            self.evaluator = GeminiGPTEvaluator()
            print("Using CROSS-MODEL ensemble (Gemini + GPT, both must agree)")
        elif use_ensemble:
            self.evaluator = GeminiEnsembleEvaluator()
            print("Using ENSEMBLE evaluator (3 prompts with unanimous voting)")
        elif use_evidence:
            self.evaluator = GeminiEvidenceEvaluator()
            print("Using EVIDENCE-BASED evaluator (Judge 2 - best performer)")
        else:
            self.evaluator = GeminiEvaluator()
            print("Using standard Gemini evaluator")

    def _evaluate_single_task(self, task_data: Dict, task_num: int, total_tasks: int) -> Dict:
        """Evaluate a single task (used for parallel execution)."""
        print(f"\n{'='*60}")
        print(f"Task {task_num}/{total_tasks}: {task_data['task_id']}")
        print(f"{'='*60}")

        try:
            result = self.evaluator.evaluate_task(task_data)
            return result
        except Exception as e:
            print(f"  ERROR: {e}")
            return {
                "task_id": task_data["task_id"],
                "run_name": task_data["run_name"],
                "task_level": task_data["task_level"],
                "error": str(e),
                "pass": False,
            }

    def run_evaluation(self, limit: int = None, output_filename: str = None, max_workers: int = 1) -> Dict:
        """
        Run evaluation on all collected tasks.
        Saves results incrementally after each task.

        Args:
            limit: Optional limit on number of tasks to evaluate (for testing)
            output_filename: Output filename for incremental saves
            max_workers: Number of parallel workers (default: 1 for sequential)

        Returns:
            Dictionary with all results and summary statistics
        """
        # Collect tasks
        print("Collecting tasks...")
        tasks = self.collector.collect_all_tasks()
        print(f"Collected {len(tasks)} tasks\n")

        if limit:
            tasks = tasks[:limit]
            print(f"Limiting to {limit} tasks for testing\n")

        # Determine output filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"evaluation_results_{timestamp}.json"

        output_path = self.output_dir / output_filename

        print(f"Running with {max_workers} parallel worker(s)\n")

        # Thread-safe lock for saving results
        save_lock = threading.Lock()
        results = []
        completed_count = [0]  # Use list to make it mutable in closure

        def save_progress():
            """Thread-safe progress saving."""
            with save_lock:
                summary = self._calculate_summary(results)
                evaluation_results = {
                    "timestamp": datetime.now().isoformat(),
                    "num_tasks_evaluated": len(results),
                    "num_tasks_total": len(tasks),
                    "summary": summary,
                    "results": results,
                }

                with open(output_path, 'w') as f:
                    json.dump(evaluation_results, f, indent=2)

                print(f"\n  [{completed_count[0]}/{len(tasks)}] Progress saved to: {output_path}")

        if max_workers == 1:
            # Sequential execution (original behavior)
            for i, task_data in enumerate(tasks, 1):
                result = self._evaluate_single_task(task_data, i, len(tasks))
                results.append(result)
                completed_count[0] = i
                save_progress()
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._evaluate_single_task, task_data, i, len(tasks)): (i, task_data)
                    for i, task_data in enumerate(tasks, 1)
                }

                # Process results as they complete
                for future in as_completed(future_to_task):
                    task_num, task_data = future_to_task[future]
                    result = future.result()

                    with save_lock:
                        results.append(result)
                        completed_count[0] += 1

                    save_progress()

        # Final summary
        summary = self._calculate_summary(results)

        # Package final results
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "num_tasks_evaluated": len(results),
            "num_tasks_total": len(tasks),
            "summary": summary,
            "results": results,
        }

        return evaluation_results

    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics from results."""
        total = len(results)
        passed = sum(1 for r in results if r.get("pass", False))
        failed = total - passed

        # By run
        by_run = {}
        for result in results:
            run_name = result["run_name"]
            if run_name not in by_run:
                by_run[run_name] = {"total": 0, "passed": 0, "failed": 0}

            by_run[run_name]["total"] += 1
            if result.get("pass", False):
                by_run[run_name]["passed"] += 1
            else:
                by_run[run_name]["failed"] += 1

        # Add pass rate
        for run_name in by_run:
            total = by_run[run_name]["total"]
            passed = by_run[run_name]["passed"]
            by_run[run_name]["pass_rate"] = passed / total if total > 0 else 0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "by_run": by_run,
        }

    def save_results(self, evaluation_results: Dict, filename: str = None):
        """Save final evaluation results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Final results saved to: {output_path}")
        print(f"{'='*60}")

        return output_path

    def print_summary(self, evaluation_results: Dict):
        """Print summary of evaluation results."""
        summary = evaluation_results["summary"]

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total tasks: {summary['total']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        print(f"Failed: {summary['failed']}")

        print(f"\nBy Run:")
        for run_name, stats in summary["by_run"].items():
            print(f"  {run_name}:")
            print(f"    Total: {stats['total']}")
            print(f"    Passed: {stats['passed']} ({stats['pass_rate']:.1%})")
            print(f"    Failed: {stats['failed']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini-based state evolution evaluation"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Path to project root directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: evaluation_results_<timestamp>.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 for sequential)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble mode with 3 prompts and unanimous voting (slower but more accurate)",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Use evidence-based prompt (Judge 2 - best single judge from ensemble)",
    )
    parser.add_argument(
        "--cross-model",
        action="store_true",
        help="Use cross-model ensemble (Gemini + GPT, both must agree)",
    )
    parser.add_argument(
        "--gpt",
        action="store_true",
        help="Use GPT-5.2 evaluator with frame extraction (every 5th frame)",
    )
    parser.add_argument(
        "--double-ensemble",
        action="store_true",
        help="Use double ensemble: each model (Gemini + GPT) uses 3 prompts with majority vote, then unanimous cross-model",
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        default="benchmark/tasks_v4",
        help="Task directory relative to project root (default: benchmark/tasks_v4)",
    )

    args = parser.parse_args()

    # Check for API keys
    if not args.gpt and not args.double_ensemble and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set!")
        print("Please set it with: export GOOGLE_API_KEY='your-api-key'")
        return

    if (args.cross_model or args.gpt or args.double_ensemble) and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return

    if args.double_ensemble and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set!")
        print("Double ensemble requires both Gemini and OpenAI API keys.")
        return

    # Run evaluation (saves incrementally)
    runner = EvaluationRunner(
        args.project_root,
        use_ensemble=args.ensemble,
        use_evidence=args.evidence,
        use_cross_model=args.cross_model,
        use_gpt=args.gpt,
        use_double_ensemble=args.double_ensemble,
        task_dir=args.task_dir
    )
    results = runner.run_evaluation(
        limit=args.limit,
        output_filename=args.output,
        max_workers=args.workers
    )

    # Final save (already done incrementally, but update one more time)
    output_path = runner.save_results(results, filename=args.output)

    # Print summary
    runner.print_summary(results)

    print(f"\nDone! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
