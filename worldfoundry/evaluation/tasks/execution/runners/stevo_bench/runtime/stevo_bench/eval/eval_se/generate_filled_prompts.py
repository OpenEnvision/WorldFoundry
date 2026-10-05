#!/usr/bin/env python3
"""
Extract prompt1 template and fill it with occlusion focus and expected process
from evaluation results.
"""

import json
from pathlib import Path


def get_occlusion_context_and_focus(model_type: str):
    """
    Determine occlusion context and focus based on model type.
    This matches the logic from gemini_evaluator_ensemble.py
    """
    if model_type in ["veo", "sora"]:
        occlusion_context = """
IMPORTANT: This video includes an OCCLUSION where an object blocks the view of the scene.
You must specifically check: Did the expected process CONTINUE HAPPENING during the occlusion (when the view was blocked)?
This is critical - the process should evolve even when we cannot see it."""
        occlusion_focus = "during the occlusion or camera turn, when the object is completely invisible"
    elif model_type in ["lingbot", "hunyuan", "genie", "wan22"]:
        occlusion_context = """
IMPORTANT: This video includes CAMERA MOVEMENT where the camera turns away from the main object.
You must specifically check: Did the expected process HAPPEN while the camera was turned away (when the main object was not visible)?
This is critical - the process should evolve even when the camera is not looking at it."""
        occlusion_focus = "during the occlusion or camera turn, when the object is completely invisible"
    else:
        occlusion_context = ""
        occlusion_focus = "in the video"

    return occlusion_context, occlusion_focus


def get_prompt1_template():
    """
    Return the prompt1 template from gemini_evaluator_ensemble.py
    """
    return """You are a CONSERVATIVE and SKEPTICAL video evaluator. Your job is to verify whether a video shows an expected physical process.

The expected process is:
{expected_process}
{occlusion_context}

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


def main():
    # Load evaluation results
    results_path = Path("eval_se/results/evaluation_results_20260226_170228_unanimous.json")

    print(f"Loading evaluation results from {results_path}...")
    with open(results_path, 'r') as f:
        eval_data = json.load(f)

    # Get prompt1 template
    prompt1_template = get_prompt1_template()

    # Process each result
    filled_prompts = []

    for result in eval_data['results']:
        task_id = result['task_id']
        model_type = result['model_type']
        expected_process_concise = result['expected_process_concise']

        # Get occlusion context and focus for this model type
        occlusion_context, occlusion_focus = get_occlusion_context_and_focus(model_type)

        # Fill in the prompt
        filled_prompt = prompt1_template.format(
            expected_process=expected_process_concise,
            occlusion_context=occlusion_context
        )

        # Create entry with all relevant information
        entry = {
            "task_id": task_id,
            "run_name": result['run_name'],
            "model_type": model_type,
            "expected_process_concise": expected_process_concise,
            "occlusion_focus": occlusion_focus,
            "occlusion_context": occlusion_context,
            "filled_prompt1": filled_prompt
        }

        filled_prompts.append(entry)

    # Write output
    output_path = Path("eval_se/results/filled_prompts1.json")

    output_data = {
        "timestamp": eval_data['timestamp'],
        "num_tasks": len(filled_prompts),
        "source_file": str(results_path),
        "prompts": filled_prompts
    }

    print(f"Writing {len(filled_prompts)} filled prompts to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Done! Generated {len(filled_prompts)} filled prompts.")
    print(f"\nExample (first task):")
    print(f"Task ID: {filled_prompts[0]['task_id']}")
    print(f"Model Type: {filled_prompts[0]['model_type']}")
    print(f"Occlusion Focus: {filled_prompts[0]['occlusion_focus']}")
    print(f"\nFilled Prompt (truncated):")
    print(filled_prompts[0]['filled_prompt1'][:300] + "...")


if __name__ == "__main__":
    main()
