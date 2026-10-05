"""
Double ensemble evaluator:
- Each model (Gemini and GPT) uses 3 prompts with majority voting
- Final verdict requires BOTH models to agree (unanimous)

Gemini: 3 prompts → majority vote → Gemini verdict
GPT: 3 prompts → majority vote → GPT verdict
Final: Gemini verdict AND GPT verdict must both be True
"""

import os
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import cv2

# Suppress gRPC fork warnings when using parallel execution
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import google.generativeai as genai
from PIL import Image
from openai import OpenAI


class GeminiGPTDoubleEnsemble:
    def __init__(self, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """Initialize double ensemble evaluator with both API keys."""
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

        self.gpt_model = "gpt-5.2"

        print(f"Using DOUBLE ENSEMBLE:")
        print(f"  - Gemini: 3 prompts → majority vote")
        print(f"  - GPT: 3 prompts → majority vote")
        print(f"  - Final: Both must agree (unanimous)")

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _extract_frames(self, video_path: str, frame_interval: int = 5) -> List[str]:
        """Extract frames from video at regular intervals."""
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
        """Use Gemini to predict expected process (single call, not ensemble)."""
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

        # Call Gemini
        print(f"    Calling Gemini for process prediction...")
        try:
            image = Image.open(image_path)
            response = self.gemini_model.generate_content([prompt, image])
            elapsed = time.time() - start_time
            print(f"    Process prediction took {elapsed:.1f}s")

            full_response = response.text
            concise_process = self._parse_process_line(full_response)

            return full_response, concise_process
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"    ERROR after {elapsed:.1f}s: {e}")
            raise

    def _parse_process_line(self, response_text: str) -> str:
        """Parse the concise process description from the "Process:" line."""
        lines = response_text.strip().split("\n")
        for line in reversed(lines):
            line_stripped = line.strip()
            if line_stripped.lower().startswith("process:"):
                return line_stripped[8:].strip()

        print("    Warning: Could not find 'Process:' line, using full response")
        return response_text

    def _get_verification_prompts(
        self, expected_process: str, model_type: str
    ) -> List[str]:
        """Generate 3 different prompt variations for ensemble voting."""
        # Determine occlusion question based on model type
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

        # Prompt 1: Conservative/Skeptical
        prompt1 = f"""You are a CONSERVATIVE and SKEPTICAL video evaluator. Your job is to verify whether a video shows an expected physical process.

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

        # Prompt 2: Evidence-based
        prompt2 = f"""You are evaluating a video to determine if a specific physical process occurred.

Expected process:
{expected_process}
{occlusion_context}

Your task is to provide EVIDENCE-BASED evaluation:
1. List OBSERVABLE states BEFORE the occlusion/camera movement
2. List OBSERVABLE states AFTER the occlusion/camera movement
3. Based on these observations, did the expected process occur {occlusion_focus}?
4. What specific visual evidence supports your conclusion?

CRITICAL: Only answer "Yes" if you can clearly observe state changes that prove the process happened.
If you cannot observe clear evidence of the process, answer "No".

Please ignore physical violations or artifacts.

Format your response as:
VERDICT: [Yes/No]
EXPLANATION: [List your observations and evidence]"""

        # Prompt 3: Comparative/Analytical
        prompt3 = f"""You are analyzing whether a video demonstrates an expected physical process.

Expected process:
{expected_process}
{occlusion_context}

Analytical evaluation steps:
1. What is the initial state of the main object (before occlusion/camera turn)?
2. What is the final state of the main object (after it becomes visible again)?
3. Does the change from initial to final state match the expected process?
4. Could this change have occurred {occlusion_focus}?

Answer "Yes" ONLY if:
- The final state clearly differs from the initial state
- The change is consistent with the expected process
- The timing suggests it happened during the unseen period

Answer "No" if the states are unchanged or if change is inconsistent with expectations.

Please ignore physical violations or artifacts.

Format your response as:
VERDICT: [Yes/No]
EXPLANATION: [Your analysis]"""

        return [prompt1, prompt2, prompt3]

    def _verify_with_gemini_ensemble(
        self, video_path: str, expected_process: str, model_type: str
    ) -> Tuple[Dict, bool]:
        """Run Gemini 3-prompt ensemble with majority voting."""
        # Upload video file
        upload_start = time.time()
        print(f"    Uploading video for Gemini...")
        video_file = genai.upload_file(path=video_path)
        upload_time = time.time() - upload_start
        print(f"    Upload took {upload_time:.1f}s")

        # Wait for video processing
        print(f"    Waiting for video processing...")
        wait_count = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            wait_count += 1
            if wait_count % 5 == 0:
                print(f"    Still processing... ({wait_count}s)")
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed")

        # Get 3 prompt variations
        prompts = self._get_verification_prompts(expected_process, model_type)

        # Run all 3 prompts
        results = []
        verdicts = []

        for i, prompt in enumerate(prompts, 1):
            print(f"    Gemini prompt {i}/3...")
            verify_start = time.time()

            try:
                response = self.gemini_model.generate_content([video_file, prompt])
                response_text = response.text
                verify_time = time.time() - verify_start

                verdict = self._parse_verdict(response_text)
                results.append({
                    "prompt_id": i,
                    "response": response_text,
                    "verdict": verdict,
                    "time": verify_time
                })
                verdicts.append(verdict)

                print(f"      Verdict {i}: {'PASS' if verdict else 'FAIL'} ({verify_time:.1f}s)")

            except Exception as e:
                print(f"      ERROR in prompt {i}: {e}")
                results.append({
                    "prompt_id": i,
                    "response": f"ERROR: {str(e)}",
                    "verdict": None,
                    "time": time.time() - verify_start
                })

        # Majority voting
        valid_verdicts = [v for v in verdicts if v is not None]

        if not valid_verdicts:
            print("    Gemini: All prompts failed")
            final_verdict = False
        else:
            vote_counts = Counter(valid_verdicts)
            final_verdict = vote_counts.most_common(1)[0][0]
            yes_votes = sum(1 for v in valid_verdicts if v)
            no_votes = sum(1 for v in valid_verdicts if not v)
            print(f"    Gemini ensemble: Yes={yes_votes}, No={no_votes} → {'PASS' if final_verdict else 'FAIL'}")

        return {
            "individual_results": results,
            "verdicts": verdicts,
            "final_verdict": final_verdict,
            "vote_counts": dict(Counter(valid_verdicts)) if valid_verdicts else {}
        }, final_verdict

    def _verify_with_gpt_ensemble(
        self, frame_paths: List[str], expected_process: str, model_type: str
    ) -> Tuple[Dict, bool]:
        """Run GPT 3-prompt ensemble with majority voting using frames."""
        # Get 3 prompt variations
        prompts = self._get_verification_prompts(expected_process, model_type)

        # Run all 3 prompts
        results = []
        verdicts = []

        for i, prompt in enumerate(prompts, 1):
            print(f"    GPT prompt {i}/3...")
            verify_start = time.time()

            try:
                # Create new client for thread safety
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                # Build message content with frames
                content = [{"type": "text", "text": f"{prompt}\n\nYou are given a sequence of frames from the video (every 5th frame). Watch the progression carefully."}]

                for frame_path in frame_paths:
                    base64_frame = self._encode_image(frame_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_frame}"
                        }
                    })

                # Call GPT
                response = client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=1000
                )

                response_text = response.choices[0].message.content
                verify_time = time.time() - verify_start

                verdict = self._parse_verdict(response_text)
                results.append({
                    "prompt_id": i,
                    "response": response_text,
                    "verdict": verdict,
                    "time": verify_time
                })
                verdicts.append(verdict)

                print(f"      Verdict {i}: {'PASS' if verdict else 'FAIL'} ({verify_time:.1f}s)")

            except Exception as e:
                print(f"      ERROR in prompt {i}: {e}")
                results.append({
                    "prompt_id": i,
                    "response": f"ERROR: {str(e)}",
                    "verdict": None,
                    "time": time.time() - verify_start
                })

        # Majority voting
        valid_verdicts = [v for v in verdicts if v is not None]

        if not valid_verdicts:
            print("    GPT: All prompts failed")
            final_verdict = False
        else:
            vote_counts = Counter(valid_verdicts)
            final_verdict = vote_counts.most_common(1)[0][0]
            yes_votes = sum(1 for v in valid_verdicts if v)
            no_votes = sum(1 for v in valid_verdicts if not v)
            print(f"    GPT ensemble: Yes={yes_votes}, No={no_votes} → {'PASS' if final_verdict else 'FAIL'}")

        return {
            "individual_results": results,
            "verdicts": verdicts,
            "final_verdict": final_verdict,
            "vote_counts": dict(Counter(valid_verdicts)) if valid_verdicts else {}
        }, final_verdict

    def verify_process_in_video_double_ensemble(
        self, video_path: str, expected_process: str, model_type: str = "unknown"
    ) -> Tuple[Dict, bool]:
        """
        Double ensemble verification:
        1. Gemini: 3 prompts → majority vote
        2. GPT: 3 prompts → majority vote
        3. Final: Both must agree (unanimous)
        """
        # Extract frames first (for GPT)
        print(f"    Extracting frames for GPT...")
        frame_paths = self._extract_frames(video_path, frame_interval=5)
        print(f"    Extracted {len(frame_paths)} frames")

        # Run both ensembles in parallel
        print(f"    Running both model ensembles (in parallel)...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            gemini_future = executor.submit(
                self._verify_with_gemini_ensemble, video_path, expected_process, model_type
            )
            gpt_future = executor.submit(
                self._verify_with_gpt_ensemble, frame_paths, expected_process, model_type
            )

            # Collect results
            gemini_results, gemini_verdict = gemini_future.result()
            gpt_results, gpt_verdict = gpt_future.result()

        # Clean up frames
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

        # Final unanimous voting
        consensus_verdict = gemini_verdict and gpt_verdict

        print(f"    FINAL CONSENSUS: Gemini={'PASS' if gemini_verdict else 'FAIL'}, GPT={'PASS' if gpt_verdict else 'FAIL'} → {'PASS' if consensus_verdict else 'FAIL'}")

        return {
            "gemini_ensemble": gemini_results,
            "gpt_ensemble": gpt_results,
            "gemini_verdict": gemini_verdict,
            "gpt_verdict": gpt_verdict,
            "consensus_verdict": consensus_verdict
        }, consensus_verdict

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
        Evaluate a single task with double ensemble.

        Returns:
            Dictionary with evaluation results from both model ensembles
        """
        print(f"\nEvaluating task: {task_data['task_id']} (DOUBLE ENSEMBLE)")
        print(f"  Run: {task_data['run_name']}")

        # Determine model type
        model_type = self._get_model_type(task_data['run_name'])
        print(f"  Model type: {model_type}")

        # Step 1: Predict expected process
        print("  Step 1: Predicting expected process...")
        full_prediction, concise_process = self.predict_expected_process(
            task_data["init_image_path"], task_data.get("action_prompt")
        )
        print(f"  Expected process: {concise_process[:100]}...")

        # Step 2: Double ensemble verification
        print("  Step 2: Double ensemble verification...")
        ensemble_results, consensus_verdict = self.verify_process_in_video_double_ensemble(
            task_data["video_path"], concise_process, model_type
        )
        print(f"  Final Verdict: {'PASS' if consensus_verdict else 'FAIL'}")

        # Compile results
        return {
            "task_id": task_data["task_id"],
            "run_name": task_data["run_name"],
            "model_type": model_type,
            "task_level": task_data["task_level"],
            "action_prompt": task_data.get("action_prompt"),
            "expected_process_full": full_prediction,
            "expected_process_concise": concise_process,
            "double_ensemble_results": ensemble_results,
            "verdict": consensus_verdict,
            "pass": consensus_verdict,
        }


if __name__ == "__main__":
    # Test the evaluator
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gemini_gpt_double_ensemble.py <path_to_project_root>")
        sys.exit(1)

    from data_collector import TaskDataCollector

    project_root = sys.argv[1]
    collector = TaskDataCollector(project_root)
    tasks = collector.collect_all_tasks()

    if not tasks:
        print("No tasks found!")
        sys.exit(1)

    # Test on first task
    evaluator = GeminiGPTDoubleEnsemble()
    result = evaluator.evaluate_task(tasks[0])

    print("\n" + "=" * 60)
    print("DOUBLE ENSEMBLE RESULT")
    print("=" * 60)
    for key, value in result.items():
        if key == "double_ensemble_results":
            print(f"{key}:")
            print(f"  Gemini verdict: {value['gemini_verdict']}")
            print(f"  GPT verdict: {value['gpt_verdict']}")
            print(f"  Consensus: {value['consensus_verdict']}")
        elif key in ["expected_process_full"]:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
