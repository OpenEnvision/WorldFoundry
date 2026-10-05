"""
Recompute ensemble results using unanimous voting instead of majority voting.
Uses existing evaluation results with saved individual judge verdicts.
"""

import json
import sys
from pathlib import Path
from typing import Dict


def recompute_with_unanimous(results_file: str, project_root: str) -> Dict:
    """
    Recompute ensemble verdicts using unanimous voting.

    Args:
        results_file: Path to ensemble evaluation results JSON
        project_root: Project root directory

    Returns:
        Dictionary with recomputed results and comparison stats
    """
    # Load existing results
    results_path = Path(results_file)
    if not results_path.is_absolute():
        results_path = Path(project_root) / "eval_se" / "results" / results_file

    with open(results_path, 'r') as f:
        eval_results = json.load(f)

    # Check if this is ensemble results
    if not eval_results["results"] or "ensemble_results" not in eval_results["results"][0]:
        print("ERROR: This file does not contain ensemble results!")
        sys.exit(1)

    # Recompute verdicts
    changes = []
    unanimous_results = []

    majority_passes = 0
    unanimous_passes = 0

    for result in eval_results["results"]:
        task_id = result["task_id"]
        run_name = result["run_name"]

        # Get individual judge verdicts
        ensemble_data = result["ensemble_results"]
        verdicts = ensemble_data["verdicts"]

        # Filter out None (failed API calls)
        valid_verdicts = [v for v in verdicts if v is not None]

        # Original verdict (majority voting)
        original_verdict = result["verdict"]

        # Recompute with unanimous voting
        if not valid_verdicts:
            unanimous_verdict = False
        else:
            unanimous_verdict = all(v == True for v in valid_verdicts)

        # Track passes
        if original_verdict:
            majority_passes += 1
        if unanimous_verdict:
            unanimous_passes += 1

        # Create new result entry
        new_result = result.copy()
        new_result["verdict"] = unanimous_verdict
        new_result["pass"] = unanimous_verdict
        new_result["original_verdict_majority"] = original_verdict

        unanimous_results.append(new_result)

        # Track changes
        if original_verdict != unanimous_verdict:
            yes_votes = sum(1 for v in valid_verdicts if v)
            no_votes = sum(1 for v in valid_verdicts if not v)

            changes.append({
                "task_id": task_id,
                "run_name": run_name,
                "majority_verdict": original_verdict,
                "unanimous_verdict": unanimous_verdict,
                "individual_verdicts": verdicts,
                "yes_votes": yes_votes,
                "no_votes": no_votes
            })

    # Create new results file
    new_eval_results = eval_results.copy()
    new_eval_results["results"] = unanimous_results
    new_eval_results["voting_method"] = "unanimous"
    new_eval_results["recomputed_from"] = str(results_path)

    return {
        "new_results": new_eval_results,
        "changes": changes,
        "stats": {
            "total_tasks": len(eval_results["results"]),
            "majority_passes": majority_passes,
            "unanimous_passes": unanimous_passes,
            "changed_verdicts": len(changes),
            "majority_pass_rate": majority_passes / len(eval_results["results"]),
            "unanimous_pass_rate": unanimous_passes / len(eval_results["results"])
        }
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python recompute_unanimous.py <ensemble_results_file.json>")
        print("Example: python recompute_unanimous.py evaluation_results_20260226_161252.json")
        sys.exit(1)

    results_file = sys.argv[1]
    project_root = "."

    print("Recomputing ensemble results with unanimous voting...")
    recomputed = recompute_with_unanimous(results_file, project_root)

    stats = recomputed["stats"]
    changes = recomputed["changes"]

    print(f"\n{'='*70}")
    print("UNANIMOUS VOTING RECOMPUTATION")
    print(f"{'='*70}")
    print(f"Total tasks evaluated: {stats['total_tasks']}")
    print(f"Majority voting passes: {stats['majority_passes']} ({stats['majority_pass_rate']*100:.1f}%)")
    print(f"Unanimous voting passes: {stats['unanimous_passes']} ({stats['unanimous_pass_rate']*100:.1f}%)")
    print(f"Changed verdicts: {stats['changed_verdicts']}")

    if changes:
        print(f"\n{'='*70}")
        print(f"CHANGED VERDICTS ({len(changes)} tasks)")
        print(f"{'='*70}")

        # Group by change type
        majority_to_fail = [c for c in changes if c["majority_verdict"] and not c["unanimous_verdict"]]
        fail_to_majority = [c for c in changes if not c["majority_verdict"] and c["unanimous_verdict"]]

        print(f"\nMajority PASS → Unanimous FAIL: {len(majority_to_fail)}")
        if majority_to_fail:
            print("(These had split votes, now require unanimous agreement)")
            for c in majority_to_fail[:10]:  # Show first 10
                print(f"  - {c['task_id']} ({c['run_name']}): Votes={c['yes_votes']}-Yes {c['no_votes']}-No")
            if len(majority_to_fail) > 10:
                print(f"  ... and {len(majority_to_fail) - 10} more")

        print(f"\nMajority FAIL → Unanimous PASS: {len(fail_to_majority)}")
        if fail_to_majority:
            print("(These should not happen with unanimous voting - check logic)")
            for c in fail_to_majority:
                print(f"  - {c['task_id']} ({c['run_name']}): Votes={c['yes_votes']}-Yes {c['no_votes']}-No")

    # Save new results
    input_stem = Path(results_file).stem
    output_file = f"{input_stem}_unanimous.json"
    output_path = Path(project_root) / "eval_se" / "results" / output_file

    with open(output_path, 'w') as f:
        json.dump(recomputed["new_results"], f, indent=2)

    print(f"\n{'='*70}")
    print(f"Recomputed results saved to: {output_path}")
    print(f"\nYou can now analyze this with:")
    print(f"  python eval_se/analyze_results.py {output_file}")
    print(f"  python eval_se/model_win_rates.py {output_file}")
    print(f"  python eval_se/pairwise_agreement.py {output_file}")


if __name__ == "__main__":
    main()
