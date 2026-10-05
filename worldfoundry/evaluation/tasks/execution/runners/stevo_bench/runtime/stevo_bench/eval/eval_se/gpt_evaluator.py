"""
GPT-based evaluator for state evolution in videos.
Extracts frames (every 5th frame) and sends them to GPT-5.2.
Two-step evaluation:
1. Predict expected process from initial image + action prompt
2. Verify if predicted process happened in the video (using multiple frames)
"""

import os
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import cv2
from openai import OpenAI
from PIL import Image


class GPTEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize GPT evaluator with API key."""
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Set it as environment variable or pass to constructor."
                )

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5.2"
        print(f"Using model: {self.model}")

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _extract_frames(self, video_path: str, frame_interval: int = 5) -> List[str]:
        """
        Extract frames from video at regular intervals.

        Args:
            video_path: Path to video file
            frame_interval: Extract every Nth frame (default: 5)

        Returns:
            List of paths to extracted frame images
        """
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        extracted_count = 0

        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract every Nth frame
                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(temp_dir, f"frame_{extracted_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1

                frame_count += 1

        finally:
            cap.release()

        print(f"    Extracted {extracted_count} frames from {frame_count} total frames")
        return frame_paths

    def predict_expected_process(
        self, image_path: str, action_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        First GPT call: Given initial image and optional action prompt,
        predict what process should happen.

        Args:
            image_path: Path to initial image
            action_prompt: Optional action/camera movement prompt

        Returns:
            Tuple of (full_prediction, concise_process)
        """
        start_time = time.time()

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

        # Encode image
        base64_image = self._encode_image(image_path)

        # Make API call
        print(f"    Calling GPT API for process prediction...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=500
            )
            elapsed = time.time() - start_time
            print(f"    Process prediction took {elapsed:.1f}s")

            full_response = response.choices[0].message.content
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
        Second GPT call: Given video frames and expected process,
        verify if the process actually happened.

        Args:
            video_path: Path to video file
            expected_process: The expected process from first call
            model_type: Type of model (veo, sora, lingbot, hunyuan) to customize prompt

        Returns:
            Tuple of (GPT's response, boolean indicating if process happened)
        """
        # Extract frames
        extract_start = time.time()
        print(f"    Extracting frames from video...")
        frame_paths = self._extract_frames(video_path, frame_interval=5)
        extract_time = time.time() - extract_start
        print(f"    Frame extraction took {extract_time:.1f}s")

        # Construct prompt based on model type
        if model_type in ["veo", "sora"]:
            occlusion_question = """
IMPORTANT: This video includes an OCCLUSION where an object blocks the view of the scene.
You must specifically check: Did the expected process CONTINUE HAPPENING during the occlusion (when the view was blocked)?
This is critical - the process should evolve even when we cannot see it."""
        elif model_type in ["lingbot", "hunyuan", "genie", "wan22"]:
            occlusion_question = """
IMPORTANT: This video includes CAMERA MOVEMENT where the camera turns away from the main object.
You must specifically check: Did the expected process HAPPEN while the camera was turned away (when the main object was not visible)?
This is critical - the process should evolve even when the camera is not looking at it."""
        else:
            occlusion_question = ""

        prompt = f"""You are a CONSERVATIVE and SKEPTICAL video evaluator. Your job is to verify whether a video shows an expected physical process.

The expected process is:
{expected_process}
{occlusion_question}

You are given a sequence of frames from the video. Watch the progression carefully and answer:
1. Did the expected process happen in the video? (Yes/No)
2. Provide a detailed explanation of what is the object state right before it becomes invisible, and right after it is seen again.
3. Compare what happened in the video to what was expected
4. SPECIFICALLY address: Did the process continue/happen during the occlusion or camera movement?
5. If the process did not happen as expected, explain what was different or missing

Please ignore physical violations or artifacts. Only focus on whether the key process happened DURING the occlusion or camera movement.

Format your response as:
VERDICT: [Yes/No]
EXPLANATION: [Your detailed explanation]"""

        # Build message content with all frames
        content = [{"type": "text", "text": prompt}]

        for frame_path in frame_paths:
            base64_frame = self._encode_image(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_frame}"
                }
            })

        # Make API call
        print(f"    Calling GPT API for video verification ({len(frame_paths)} frames)...")
        verify_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=1000
            )
            response_text = response.choices[0].message.content
            verify_time = time.time() - verify_start
            print(f"    Verification took {verify_time:.1f}s")

            # Parse verdict
            verdict = self._parse_verdict(response_text)

            # Clean up temp frames
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            # Remove temp directory
            try:
                temp_dir = os.path.dirname(frame_paths[0])
                os.rmdir(temp_dir)
            except:
                pass

            return response_text, verdict
        except Exception as e:
            verify_time = time.time() - verify_start
            print(f"    ERROR after {verify_time:.1f}s: {e}")
            # Clean up on error
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            raise

    def _parse_verdict(self, response_text: str) -> bool:
        """
        Parse the verdict from GPT's response.

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
        elif "genie" in run_lower:
            return "genie"
        elif "wan" in run_lower:
            return "wan22"
        else:
            return "unknown"

    def evaluate_task(self, task_data: Dict) -> Dict:
        """
        Evaluate a single task with two-step GPT evaluation.

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
        print("Usage: python gpt_evaluator.py <path_to_project_root>")
        sys.exit(1)

    from data_collector import TaskDataCollector

    project_root = sys.argv[1]
    collector = TaskDataCollector(project_root)
    tasks = collector.collect_all_tasks()

    if not tasks:
        print("No tasks found!")
        sys.exit(1)

    # Test on first task
    evaluator = GPTEvaluator()
    result = evaluator.evaluate_task(tasks[0])

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)
    for key, value in result.items():
        if key in ["expected_process_full", "verification_response"]:
            print(f"{key}:")
            print(f"  {value[:200]}...")
        else:
            print(f"{key}: {value}")
