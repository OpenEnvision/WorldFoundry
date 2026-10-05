"""
Cross-model ensemble evaluator using both Gemini and GPT.
Uses the same prompts as gemini_evaluator.py.
Sends requests to both APIs in parallel for speed.
- Gemini: Analyzes full video via API upload
- GPT: Analyzes extracted frames (every 5th frame)
Only marks as PASS if BOTH models agree.
"""

import os
import time
import base64
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

# Suppress gRPC fork warnings when using parallel execution
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import google.generativeai as genai
from PIL import Image
from openai import OpenAI


class GeminiGPTEvaluator:
    # Class-level semaphores to limit concurrent API calls
    _gemini_semaphore = threading.Semaphore(20)  # Max 20 concurrent Gemini calls
    _gpt_semaphore = threading.Semaphore(3)      # Max 3 concurrent GPT calls

    def __init__(self, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """Initialize evaluator with both Gemini and OpenAI API keys."""
        # Initialize Gemini
        if gemini_api_key is None:
            gemini_api_key = os.environ.get("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Set it as environment variable or pass to constructor."
                )

        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-3-pro-preview")

        # Initialize OpenAI
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Set it as environment variable or pass to constructor."
                )

        self.openai_client = OpenAI(api_key=openai_api_key)
        self.gpt_model = "gpt-5.2"  # GPT-5.2 with frame extraction

        print(f"Using models: gemini-3-pro-preview (full video) + {self.gpt_model} (frame extraction)")
        print(f"Rate limits: Gemini max 20 concurrent, GPT max 3 concurrent")

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

        return frame_paths

    def predict_expected_process(
        self, image_path: str, action_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        First call: Gemini predicts expected process.

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

        print(f"    Calling Gemini API for process prediction...")

        with self._gemini_semaphore:  # Limit concurrent Gemini calls
            try:
                image = Image.open(image_path)
                gemini_response = self.gemini_model.generate_content([prompt, image])
                gemini_text = gemini_response.text
                elapsed = time.time() - start_time
                print(f"    Gemini prediction took {elapsed:.1f}s")

                # Parse the concise process description
                concise_process = self._parse_process_line(gemini_text)

                return gemini_text, concise_process
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"    Gemini ERROR after {elapsed:.1f}s: {e}")
                raise

    def _parse_process_line(self, response_text: str) -> str:
        """Parse the concise process description from the "Process:" line."""
        if response_text.startswith("ERROR:"):
            return "Error in prediction"

        lines = response_text.strip().split("\n")
        for line in reversed(lines):
            line_stripped = line.strip()
            if line_stripped.lower().startswith("process:"):
                return line_stripped[8:].strip()

        print("    Warning: Could not find 'Process:' line, using full response")
        return response_text[:200]  # Truncate if too long

    def _get_gemini_verification_prompt(self, expected_process: str, model_type: str) -> str:
        """Generate Gemini-specific verification prompt."""
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

        return f"""Your job is to verify whether a video shows an expected physical process.

The expected process is:
{expected_process}
{occlusion_question}

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

    def _get_gpt_verification_prompt(self, expected_process: str, model_type: str) -> str:
        """Generate GPT-specific verification prompt (customizable)."""
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

        return f"""Your job is to verify whether a video shows an expected physical process.

The expected process is:
{expected_process}
{occlusion_question}

Watch the video carefully and answer:
1. Did the expected process happen in the video? (Yes/No)
2. Provide a detailed explanation of what is the object state right before it becomes invisible, and right after it is seen again.
3. Compare what happened in the video to what was expected
4. SPECIFICALLY address: Did the process continue/happen during the occlusion or camera movement?

Please ignore physical violations or artifacts, also ignore if trigger/occlusion are not correct. Only focus on whether the key process happened DURING the occlusion or camera movement.

Format your response as:
VERDICT: [Yes/No]
EXPLANATION: [Your detailed explanation]"""

    def verify_process_in_video(
        self, video_path: str, expected_process: str, model_type: str = "unknown"
    ) -> Tuple[Dict, bool]:
        """
        Second call: Both Gemini and GPT verify process in video.

        Returns:
            Tuple of (dict with both responses and verdicts, final consensus verdict)
        """
        # Generate separate prompts for each model
        gemini_prompt = self._get_gemini_verification_prompt(expected_process, model_type)
        gpt_prompt = self._get_gpt_verification_prompt(expected_process, model_type)

        # Extract frames first (before parallel calls)
        print(f"    Extracting frames for GPT...")
        frame_paths = self._extract_frames(video_path, frame_interval=5)

        # Call both APIs in parallel
        print(f"    Calling both APIs for video verification (in parallel)...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            gemini_future = executor.submit(self._verify_with_gemini, video_path, gemini_prompt)
            gpt_future = executor.submit(self._verify_with_gpt_frames, frame_paths, gpt_prompt)

            # Collect results as they complete
            gemini_verdict, gemini_response, gemini_time = gemini_future.result()
            gpt_verdict, gpt_response, gpt_time = gpt_future.result()

        # Clean up frames after both calls complete
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        try:
            temp_dir = os.path.dirname(frame_paths[0]) if frame_paths else None
            if temp_dir:
                os.rmdir(temp_dir)
        except:
            pass

        # Consensus: both must agree
        consensus_verdict = gemini_verdict and gpt_verdict

        print(f"    Gemini verdict: {'PASS' if gemini_verdict else 'FAIL'}")
        print(f"    GPT verdict: {'PASS' if gpt_verdict else 'FAIL'}")
        print(f"    Consensus (both agree): {'PASS' if consensus_verdict else 'FAIL'}")

        return {
            "gemini_response": gemini_response,
            "gemini_verdict": gemini_verdict,
            "gemini_time": gemini_time,
            "gpt_response": gpt_response,
            "gpt_verdict": gpt_verdict,
            "gpt_time": gpt_time,
            "consensus_verdict": consensus_verdict
        }, consensus_verdict

    def _verify_with_gemini(self, video_path: str, prompt: str) -> Tuple[bool, str, float]:
        """Verify video with Gemini."""
        with self._gemini_semaphore:  # Limit concurrent Gemini calls
            start_time = time.time()

            try:
                # Upload video
                video_file = genai.upload_file(path=video_path)

                # Wait for processing
                wait_count = 0
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    wait_count += 1
                    if wait_count % 5 == 0:
                        print(f"      Gemini still processing video... ({wait_count}s)")
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    raise ValueError(f"Gemini video processing failed")

                # Make API call
                response = self.gemini_model.generate_content([video_file, prompt])
                response_text = response.text
                verdict = self._parse_verdict(response_text)

                elapsed = time.time() - start_time
                print(f"    Gemini verification took {elapsed:.1f}s")

                return verdict, response_text, elapsed

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"    Gemini ERROR after {elapsed:.1f}s: {e}")
                return False, f"ERROR: {str(e)}", elapsed

    def _verify_with_gpt_frames(self, frame_paths: List[str], prompt: str) -> Tuple[bool, str, float]:
        """Verify video with GPT using pre-extracted frames."""
        with self._gpt_semaphore:  # Limit concurrent GPT calls
            start_time = time.time()
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    # Create a new OpenAI client for thread safety
                    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                    # Build message content with all frames
                    content = [{"type": "text", "text": f"{prompt}\n\nYou are given a sequence of frames from the video (every 5th frame). Watch the progression carefully."}]

                    for frame_path in frame_paths:
                        base64_frame = self._encode_image(frame_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frame}"
                            }
                        })

                    # Call GPT with all frames
                    response = client.chat.completions.create(
                        model=self.gpt_model,
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=1000
                    )

                    response_text = response.choices[0].message.content
                    verdict = self._parse_verdict(response_text)

                    elapsed = time.time() - start_time
                    print(f"    GPT verification took {elapsed:.1f}s ({len(frame_paths)} frames)")

                    return verdict, response_text, elapsed

                except Exception as e:
                    error_str = str(e)
                    # Check if it's a 400 error or rate limit error
                    if ("400" in error_str or "rate" in error_str.lower()) and attempt < max_retries - 1:
                        print(f"      GPT verification attempt {attempt + 1} failed with 400, retrying in 1s...")
                        time.sleep(1)
                        continue
                    else:
                        elapsed = time.time() - start_time
                        print(f"    GPT ERROR after {elapsed:.1f}s: {error_str}")
                        return False, f"ERROR: {error_str}", elapsed

    def _parse_verdict(self, response_text: str) -> bool:
        """Parse the verdict from response text."""
        if response_text.startswith("ERROR:"):
            return False

        lines = response_text.split("\n")
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("verdict:"):
                verdict_part = line_lower.replace("verdict:", "").strip()
                if "yes" in verdict_part:
                    return True
                elif "no" in verdict_part:
                    return False

        # Fallback
        first_100_chars = response_text[:100].lower()
        if "verdict: yes" in first_100_chars or "verdict:yes" in first_100_chars:
            return True

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
        Evaluate a single task with cross-model ensemble.

        Returns:
            Dictionary with evaluation results from both models
        """
        print(f"\nEvaluating task: {task_data['task_id']} (CROSS-MODEL ENSEMBLE)")
        print(f"  Run: {task_data['run_name']}")

        # Determine model type
        model_type = self._get_model_type(task_data['run_name'])
        print(f"  Model type: {model_type}")

        # Step 1: Predict expected process (Gemini only)
        print("  Step 1: Predicting expected process (Gemini)...")
        full_prediction, concise_process = self.predict_expected_process(
            task_data["init_image_path"], task_data.get("action_prompt")
        )
        print(f"  Expected process (concise): {concise_process[:100]}...")

        # Step 2: Verify process in video (both models)
        print("  Step 2: Verifying process in video (Gemini + GPT)...")
        verification_results, consensus_verdict = self.verify_process_in_video(
            task_data["video_path"], concise_process, model_type
        )
        print(f"  Final Consensus Verdict: {'PASS' if consensus_verdict else 'FAIL'}")

        # Compile results
        return {
            "task_id": task_data["task_id"],
            "run_name": task_data["run_name"],
            "model_type": model_type,
            "task_level": task_data["task_level"],
            "action_prompt": task_data.get("action_prompt"),
            "prediction_gemini": full_prediction,
            "expected_process_concise": concise_process,
            "verification_results": verification_results,
            "verdict": consensus_verdict,
            "pass": consensus_verdict,
        }


if __name__ == "__main__":
    # Test the evaluator
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gemini_gpt_evaluator.py <path_to_project_root>")
        sys.exit(1)

    from data_collector import TaskDataCollector

    project_root = sys.argv[1]
    collector = TaskDataCollector(project_root)
    tasks = collector.collect_all_tasks()

    if not tasks:
        print("No tasks found!")
        sys.exit(1)

    # Test on first task
    evaluator = GeminiGPTEvaluator()
    result = evaluator.evaluate_task(tasks[0])

    print("\n" + "=" * 60)
    print("CROSS-MODEL ENSEMBLE RESULT")
    print("=" * 60)
    for key, value in result.items():
        if key == "prediction_gemini":
            print(f"{key}:")
            print(f"  {value[:200]}...")
        elif key == "verification_results":
            print(f"{key}:")
            print(f"  Gemini verdict: {value['gemini_verdict']}")
            print(f"  GPT verdict: {value['gpt_verdict']}")
            print(f"  Consensus: {value['consensus_verdict']}")
        else:
            print(f"{key}: {value}")
