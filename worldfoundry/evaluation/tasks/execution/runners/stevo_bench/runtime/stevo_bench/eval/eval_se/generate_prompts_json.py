"""
Generate a JSON file with all task IDs and complete Gemini evaluation prompts.
Extracts expected_process_concise from evaluation results and generates full prompts.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def get_occlusion_text(model_type: str) -> str:
    """Get the occlusion-specific text based on model type."""
    if model_type in ["veo", "sora"]:
        return """
IMPORTANT: This video includes an OCCLUSION where an object blocks the view of the scene.
You must specifically check: Did the expected process CONTINUE HAPPENING during the occlusion (when the view was blocked)?
This is critical - the process should evolve even when we cannot see it."""
    elif model_type in ["lingbot", "hunyuan", "genie", "wan22"]:
        return """
IMPORTANT: This video includes CAMERA MOVEMENT where the camera turns away from the main object.
You must specifically check: Did the expected process HAPPEN while the camera was turned away (when the main object was not visible)?
This is critical - the process should evolve even when the camera is not looking at it."""
    else:
        return ""


def generate_gemini_prompt(expected_process: str, model_type: str) -> str:
    """Generate the complete Gemini verification prompt."""
    occlusion_text = get_occlusion_text(model_type)

    return f"""Your job is to verify whether a video shows an expected physical process.

The expected process is:
{expected_process}
{occlusion_text}

Watch the video carefully and answer:
1. Did the expected process happen in the video? (Yes/No)
2. Provide a detailed explanation of what is the object state right before it becomes invisible, and right after it is seen again.
3. Compare what happened in the video to what was expected
4. SPECIFICALLY address: Did the process continue/happen during the occlusion or camera movement?
5. If the process did not happen as expected, explain what was different or missing

Please ignore physical violations or artifacts. Only focus on whether the key process happened DURING the occlusion or camera movement.

Format your response as:
VERDICT: [Yes/No]
EXPLANATION: [Your detailed explanation]"""


def extract_prompts_from_results(results_file: str, output_file: str = None):
    """
    Extract all tasks and generate complete prompts from evaluation results.

    Args:
        results_file: Path to evaluation results JSON file
        output_file: Optional output filename
    """
    # Load results
    with open(results_file) as f:
        data = json.load(f)

    # Extract unique tasks and their info
    tasks_info = {}

    for result in data["results"]:
        task_id = result["task_id"]
        run_name = result["run_name"]
        key = f"{task_id}_{run_name}"

        if key not in tasks_info:
            tasks_info[key] = {
                "task_id": task_id,
                "run_name": run_name,
                "model_type": result.get("model_type", "unknown"),
                "task_level": result.get("task_level"),
                "action_prompt": result.get("action_prompt"),
                "expected_process_concise": result.get("expected_process_concise", "")
            }

    # Generate prompts for each task
    prompts_data = []

    for key, info in sorted(tasks_info.items()):
        # Generate complete Gemini prompt
        gemini_prompt = generate_gemini_prompt(
            info["expected_process_concise"],
            info["model_type"]
        )

        prompts_data.append({
            "task_id": info["task_id"],
            "run_name": info["run_name"],
            "model_type": info["model_type"],
            "task_level": info["task_level"],
            "action_prompt": info["action_prompt"],
            "expected_process_concise": info["expected_process_concise"],
            "gemini_verification_prompt": gemini_prompt
        })

    # Create output structure
    output = {
        "source_file": str(results_file),
        "num_tasks": len(prompts_data),
        "prompt_template": "Gemini verification prompt (step 2)",
        "tasks": prompts_data
    }

    # Determine output filename
    if output_file is None:
        output_file = "gemini_prompts_all_tasks.json"

    output_path = Path("eval_se/results") / output_file

    # Save to file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated prompts for {len(prompts_data)} tasks")
    print(f"Saved to: {output_path}")

    # Print sample
    if prompts_data:
        print(f"\nSample prompt for {prompts_data[0]['task_id']}:")
        print("="*60)
        print(prompts_data[0]['gemini_verification_prompt'][:300] + "...")

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_prompts_json.py <results_file.json> [output_file.json]")
        print("Example: python generate_prompts_json.py evaluation_results_20260226_231425.json")
        sys.exit(1)

    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Convert to Path and check if it exists
    results_path = Path(results_file)
    if not results_path.exists():
        # Try adding eval_se/results prefix
        results_path = Path("eval_se/results") / results_file
        if not results_path.exists():
            print(f"Error: Could not find {results_file}")
            sys.exit(1)

    extract_prompts_from_results(str(results_path), output_file)


if __name__ == "__main__":
    main()
