"""
Gemini-based evaluator for state evolution in videos.
Two-step evaluation:
1. Predict expected process from initial image + action prompt
2. Verify if predicted process happened in the video
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import google.generativeai as genai
from PIL import Image


class GeminiEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini evaluator with API key."""
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Set it as environment variable or pass to constructor."
                )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-3-pro-preview")
        print(f"Using model: gemini-3-pro-preview")

    def predict_expected_process(
        self, image_path: str, action_prompt: Optional[str] = None
    ) -> str:
        """
        First Gemini call: Given initial image and optional action prompt,
        predict what process should happen.

        Args:
            image_path: Path to initial image
            action_prompt: Optional action/camera movement prompt

        Returns:
            Gemini's prediction of expected process
        """
        start_time = time.time()

        # Load image
        image = Image.open(image_path)

        # Construct prompt
        if action_prompt:
            prompt = f"""You are analyzing a video generation task. You are given:
1. An initial image showing the starting state
2. An action/event that will occur: "{action_prompt}"

Based on the initial image and the action described, what physical process or state evolution should happen?
Describe the expected changes in the scene, including:
- What objects or elements will change
- How they will change (e.g., shape, position, state)
- What the final state should look like

Be specific and concrete in your description.

After your full explanation, provide a one-sentence summary on the last line in this format:
Process: [concise one-sentence description of the expected evolution]"""
        else:
            prompt = """You are analyzing a video generation task. You are given an initial image showing the starting state.

Based on what you see in the image, what physical process or state evolution appears to be set up or about to happen?
Describe the expected changes in the scene, including:
- What objects or elements will change
- How they will change (e.g., shape, position, state)
- What the final state should look like

Be specific and concrete in your description.

After your full explanation, provide a one-sentence summary on the last line in this format:
Process: [concise one-sentence description of the expected evolution]"""

        # Make API call
        print(f"    Calling Gemini API for process prediction...")
        try:
            response = self.model.generate_content([prompt, image])
            elapsed = time.time() - start_time
            print(f"    Process prediction took {elapsed:.1f}s")

            full_response = response.text
            # Parse the concise process description from "Process:" line
            concise_process = self._parse_process_line(full_response)

            return full_response, concise_process
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"    ERROR after {elapsed:.1f}s: {e}")
            raise

    def _parse_process_line(self, response_text: str) -> str:
        """
        Parse the concise process description from the "Process:" line.

        Returns:
            The concise process description, or full text if parsing fails
        """
        lines = response_text.strip().split("\n")
        for line in reversed(lines):  # Start from end to find the last Process: line
            line_stripped = line.strip()
            if line_stripped.lower().startswith("process:"):
                return line_stripped[8:].strip()  # Remove "Process:" prefix

        # Fallback: return full response if no Process: line found
        print("    Warning: Could not find 'Process:' line, using full response")
        return response_text

    def verify_process_in_video(
        self, video_path: str, expected_process: str, model_type: str = "unknown"
    ) -> Tuple[str, bool]:
        """
        Second Gemini call: Given video and expected process,
        verify if the process actually happened.

        Args:
            video_path: Path to video file
            expected_process: The expected process from first call
            model_type: Type of model (veo, sora, lingbot, hunyuan) to customize prompt

        Returns:
            Tuple of (Gemini's response, boolean indicating if process happened)
        """
        # Upload video file
        upload_start = time.time()
        print(f"    Uploading video...")
        video_file = genai.upload_file(path=video_path)
        upload_time = time.time() - upload_start
        print(f"    Upload took {upload_time:.1f}s")

        # Wait for video processing
        print(f"    Waiting for video processing...")
        process_start = time.time()
        wait_count = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            wait_count += 1
            if wait_count % 5 == 0:
                print(f"    Still processing... ({wait_count}s)")
            video_file = genai.get_file(video_file.name)

        process_time = time.time() - process_start
        print(f"    Video processing took {process_time:.1f}s")

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed for {video_path}")

        # Construct prompt based on model type
        if model_type in ["veo", "sora"]:
            occlusion_question = """
IMPORTANT: This video includes an OCCLUSION where an object blocks the view of the scene.
You must specifically check: Did the expected process CONTINUE HAPPENING during the occlusion (when the view was blocked)?
This is critical - the process should evolve even when we cannot see it."""
        elif model_type in ["lingbot", "hunyuan"]:
            occlusion_question = """
IMPORTANT: This video includes CAMERA MOVEMENT where the camera turns away from the main object.
You must specifically check: Did the expected process HAPPEN while the camera was turned away (when the main object was not visible)?
This is critical - the process should evolve even when the camera is not looking at it."""
        else:
            occlusion_question = ""

        prompt = f"""You are evaluating whether a video shows an expected physical process.

The expected process is:
{expected_process}
{occlusion_question}

Watch the video carefully and answer:
1. Did the expected process happen in the video? (Yes/No)
2. Provide a detailed explanation of what you observed in the video
3. Compare what happened in the video to what was expected
4. SPECIFICALLY address: Did the process continue/happen during the occlusion or camera movement?
5. If the process did not happen as expected, explain what was different or missing

Please only focus on whether the key process happened during the occlusion or camera movement, and ignore physical violations or artifacts.

Format your response as:
VERDICT: [Yes/No]
EXPLANATION: [Your detailed explanation]"""

        # Make API call
        print(f"    Calling Gemini API for video verification...")
        verify_start = time.time()
        response = self.model.generate_content([video_file, prompt])
        response_text = response.text
        verify_time = time.time() - verify_start
        print(f"    Verification took {verify_time:.1f}s")

        # Parse verdict
        verdict = self._parse_verdict(response_text)

        return response_text, verdict

    def _parse_verdict(self, response_text: str) -> bool:
        """
        Parse the verdict from Gemini's response.

        Returns:
            True if process happened, False otherwise
        """
        # Look for VERDICT: Yes/No in response
        lines = response_text.split("\n")
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("verdict:"):
                verdict_part = line_lower.replace("verdict:", "").strip()
                if "yes" in verdict_part:
                    return True
                elif "no" in verdict_part:
                    return False

        # Fallback: check if "yes" appears in first few lines
        first_100_chars = response_text[:100].lower()
        if "verdict: yes" in first_100_chars or "verdict:yes" in first_100_chars:
            return True

        # Default to False if unclear
        return False

    def _get_model_type(self, run_name: str) -> str:
        """Determine model type from run name."""
        run_lower = run_name.lower()
        if "veo" in run_lower:
            return "veo"
        elif "sora" in run_lower:
            return "sora"
        elif "lingbot" in run_lower:
            return "lingbot"
        elif "hunyuan" in run_lower or "hy-" in run_lower:
            return "hunyuan"
        else:
            return "unknown"

    def evaluate_task(
        self, task_data: Dict
    ) -> Dict:
        """
        Evaluate a single task with two-step Gemini evaluation.

        Args:
            task_data: Dictionary with task information including:
                - init_image_path
                - video_path
                - action_prompt (optional)
                - run_name (for determining model type)

        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating task: {task_data['task_id']}")
        print(f"  Run: {task_data['run_name']}")

        # Determine model type
        model_type = self._get_model_type(task_data['run_name'])
        print(f"  Model type: {model_type}")

        # Step 1: Predict expected process
        print("  Step 1: Predicting expected process...")
        full_prediction, concise_process = self.predict_expected_process(
            task_data["init_image_path"], task_data.get("action_prompt")
        )
        print(f"  Expected process (concise): {concise_process[:100]}...")

        # Step 2: Verify process in video (using concise process)
        print("  Step 2: Verifying process in video...")
        verification_response, verdict = self.verify_process_in_video(
            task_data["video_path"], concise_process, model_type
        )
        print(f"  Verdict: {'PASS' if verdict else 'FAIL'}")

        # Compile results
        return {
            "task_id": task_data["task_id"],
            "run_name": task_data["run_name"],
            "model_type": model_type,
            "task_level": task_data["task_level"],
            "action_prompt": task_data.get("action_prompt"),
            "expected_process_full": full_prediction,
            "expected_process_concise": concise_process,
            "verification_response": verification_response,
            "verdict": verdict,
            "pass": verdict,
        }


if __name__ == "__main__":
    # Test the evaluator
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gemini_evaluator.py <path_to_project_root>")
        sys.exit(1)

    from data_collector import TaskDataCollector

    project_root = sys.argv[1]
    collector = TaskDataCollector(project_root)
    tasks = collector.collect_all_tasks()

    if not tasks:
        print("No tasks found!")
        sys.exit(1)

    # Test on first task
    evaluator = GeminiEvaluator()
    result = evaluator.evaluate_task(tasks[0])

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)
    for key, value in result.items():
        if key in ["expected_process", "verification_response"]:
            print(f"{key}:")
            print(f"  {value[:200]}...")
        else:
            print(f"{key}: {value}")
