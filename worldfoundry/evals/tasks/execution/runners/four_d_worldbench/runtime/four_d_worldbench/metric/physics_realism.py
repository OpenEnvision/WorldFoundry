#!/usr/bin/env python3
"""
Physics reasoning evaluation script
Adapted to framework based on original gpt4o_quesgen_0827_for_vbenchsubsample.py logic
Keeping the original question generation and answering logic unchanged
"""

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added: concurrent execution
from typing import Any

from caption_method.VideoImg_caption import Keye_describe_video
from openai import OpenAI
from tqdm import tqdm

try:
    from metric.utils import load_dimension_info
except ImportError:
    from .utils import load_dimension_info

try:
    from metric_compatibility import ensure_auxiliary_info_compatibility  # noqa: F401 - Availability probe.

    COMPATIBILITY_AVAILABLE = True
except ImportError:
    COMPATIBILITY_AVAILABLE = False

warnings.filterwarnings("ignore")


class PhysicsReasoningEvaluator:
    def __init__(self, device: str = "cpu"):
        """Initialize evaluator"""
        self.device = device
        print(f"Using device: {self.device}")

        # Initialize OpenAI client (reads OPENAI_API_KEY and OPENAI_BASE_URL from env)
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        self.model_type = os.environ.get("OPENAI_MODEL", "gpt-4o")

        # Physics dimensions definition (from original file)
        self.physics_dimensions = [
            "Fundamental Physics",
            "Temporal & Spatial rationality",
            "Physical Event Reasoning",
            "Force & Motion",
            "Optical Phenomena",
            "Material Interaction & Transformation",
            "Thermal & Phase Transition",
            "Overall Description",
        ]

        # self.physics_dimensions = [
        #     "Fundamental Physics",
        #     "Object Properties & Affordances",
        #     "Spatial Reasoning",
        #     "Force & Motion",
        #     "Optical Phenomena",
        #     "Material Interaction & Transformation",
        #     "Thermal and Phase Transition",
        #     "Overall Description"
        # ]

        # Physics reasoning evaluation prompt (based on original complex_prompt_v6)
        self.physics_prompt = (
            "You are a scientist who designs diagnostic yes/no questions about short real‑world scenarios. "
            "Given: (1) a natural‑language description of a real‑world scene, and (2) an ordered list of dimensions "
            "(e.g., Fundamental Physics, Optics, Material Interaction & Transformation, Force & Motion, Thermal & Phase Transition, etc.). "
            "Returns ten(10) questions in the order provided, with one to four questions per dimension. If there are too few dimensions to ask, more questions per dimension will be required."
            "Reasoning about reasonable phenomena that should occur in the real world based on the description and posing them as questions, such as a balloon should explode when a sharp object presses into it, water should boil at above 100 degrees, cheese should melt at high temperatures, etc."
            "You should reason about what should happen in the real world, each question should be crafted to be answerable solely by inspecting the description and foucus on visible phenomenon in the description without requiring external knowledge. "
            "If the input description does not contain some problem dimensions or can not design 'yes' answer question, skip generating questions for that part and move on to the next dimension and do not invent imaginary properties. "
            "Each question must stay strictly within its assigned dimension's scope. Avoid cross‑dimension leakage. "
            "Use present tense, neutral tone, and end each question with '(yes or no)'. "
            "Return a JSON array of objects, each with: "
            "{'dimension': '<dimension name>', "
            "'auxiliary_info': ['<one to four yes/no questions>']}. "
            "Preserve the dimension order. Validate: at most 8 objects, each auxiliary_info has one to four questions, "
            "all questions are dimension‑appropriate, observable, and answer 'yes'. "
            "Here are some in‑context examples:\n\n"
            "'questions': ["
            "{'dimension': 'Material Interaction & Transformation', "
            "'auxiliary_info': ["
            "'Does the blue and yellow paints visibly exist? (yes or no)', "
            "'Does the blue and yellow paints mix visibly during the stirring process? (yes or no)', "
            "'Does the blue and yellow disappear from the mixed paint? (yes or no)', "
            "'Finally, does the paint become green? (yes or no)']},\n\n"
            "{'dimension': 'Force & Motion', "
            "'auxiliary_info': ["
            "'Does the toothpaste tube contact with hands? (yes or no)', "
            "'Does the toothpaste tube deform under stress? (yes or no)', "
            "'Does the toothpaste tube be compressed and the toothpaste be expelled out of the toothpaste tube? (yes or no)']},\n\n"
            "{'dimension': 'Thermal & Phase Transition', "
            "'auxiliary_info': ["
            "'Initially, is the river in a liquid state? (yes or no)', "
            "'Finally, does the river freeze? (yes or no)']}]\n\n"
        )

        # self.prompt = (
        #     "You are a precise yes/no caption evidence checker.\n"
        #     "Goal: maximize TRUE positives while avoiding unsupported 'yes'.\n"
        #     "Input: one caption + one yes/no question about OBSERVABLE physical facts/events/states.\n"
        #     "Decision Rules (lenient mode):\n"
        #     "1. Use ONLY the caption text (but allow paraphrase / synonyms / reordering equivalence).\n"
        #     "2. A detail is SUPPORTED if its core entities & relations appear explicitly OR via clear synonym (e.g., 'cup' vs 'mug', 'holding' vs 'grasping').\n"
        #     "3. Ignore missing minor adjectives, intensifiers, colors, counts unless the question explicitly hinges on them.\n"
        #     "4. Multi‑clause question: answer 'yes' if EVERY core clause (main subject + main action/state + essential object) is supported; tolerate benign tense shifts.\n"
        #     "5. Answer 'no' ONLY if there is a required core element absent OR contradicted.\n"
        #     "6. If partially supported but missing a CORE relation (e.g., action absent), answer 'no'.\n"
        #     "7. Default bias: prefer 'yes' when evidence is reasonably present; avoid false 'no'.\n"
        #     "8. NEVER judge real‑world plausibility—only textual support.\n"
        #     "Output EXACT JSON ONLY: {\"answer\":\"yes\"} or {\"answer\":\"no\"}."
        # )

        self.prompt = "Please answer yes or no only for the following questions according to the caption."
        # self.prompt = "You are an expert at answering questions based on descriptions of generated videos, which may contain various physically unreasonable. Please answer yes or no only for the following question according to the caption. When answering, you should carefully check whether the main objects and behaviors of the question and caption are consistent."

        # Added: control concurrency level
        self.max_workers = int(os.environ.get("PR_MAX_WORKERS", "10"))

    def generate_physics_questions(self, description: str) -> list[dict[str, Any]]:
        """
        Generate physics reasoning questions (keeping original logic)

        Args:
            description: Video description

        Returns:
            List[Dict]: List of generated questions
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an useful assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text.The format should be: questions:[q1,q2,...].",
                    },
                    {"role": "user", "content": f"prompt:{self.physics_prompt}"},
                    {"role": "user", "content": f"dimension:{self.physics_dimensions}"},
                    {"role": "user", "content": f"description:{description}"},
                ],
                temperature=0,
            )

            input_string = response.choices[0].message.content
            input_string = input_string.strip().lstrip("```json").rstrip("```").strip()  # noqa: B005 - Character-set strip of the Markdown fence; retain upstream parsing.

            try:
                result = json.loads(input_string)
                if isinstance(result, dict) and "questions" in result:
                    return result["questions"]
                elif isinstance(result, list):
                    return result
                else:
                    print(f"Unexpected response format: {result}")
                    return []
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                return []

        except Exception as e:
            print(f"Failed to generate physics questions: {e}")
            return []

    def answer_physics_question(self, question: str, caption: str) -> str:
        """
        Answer physics reasoning questions (keeping original logic)

        Args:
            caption: Video description

        Returns:
            str: Answer ('yes' / 'no')
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {
                        "role": "system",
                        "content": 'You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text. Example: {"answer":"yes"} or {"answer":"no"}',
                    },
                    {"role": "user", "content": self.prompt},
                    {"role": "user", "content": f"Caption: {caption}"},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                temperature=0,
            )

            input_string = response.choices[0].message.content
            input_string = input_string.strip().lstrip("```json").rstrip("```").strip()  # noqa: B005 - Character-set strip of the Markdown fence; retain upstream parsing.

            try:
                parsed_data = json.loads(input_string)

                if isinstance(parsed_data, dict) and "answer" in parsed_data:
                    answer_text = parsed_data["answer"].lower()
                elif isinstance(parsed_data, str):
                    answer_text = parsed_data.lower()
                else:
                    answer_text = "null"

                # Standardize answer
                if answer_text == "yes":
                    return "yes"
                elif answer_text == "no":
                    return "no"
                else:
                    return "null"

            except json.JSONDecodeError as e:
                print(f"Failed to parse answer: {e}")
                return "null"

        except Exception as e:
            print(f"Failed to answer question: {e}")
            return "null"

    # Added: extract pure question list from generated results
    def _extract_questions_from_generated(self, physics_questions_raw: list[dict[str, Any]]) -> list[str]:
        questions: list[str] = []
        for q_group in physics_questions_raw or []:
            if isinstance(q_group, dict) and "auxiliary_info" in q_group:
                questions.extend(q_group["auxiliary_info"])
        # Remove empty/non-string and deduplicate (deduplicate by lowercase trimmed trailing whitespace)
        seen = set()
        cleaned = []
        for q in questions:
            if isinstance(q, str):
                k = q.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    cleaned.append(q.strip())
        return cleaned

    # Added: ensure 10 questions, generate more if insufficient, truncate if excess
    def _ensure_ten_questions(self, prompt_text: str, base_questions: list[str]) -> list[str]:
        target = 10
        # Initial cleaning and deduplication
        uniq = []
        seen = set()
        for q in base_questions or []:
            if isinstance(q, str):
                k = q.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(q.strip())
        # If insufficient, retry up to two rounds to generate more
        retries = 0
        while len(uniq) < target and retries < 2:
            retries += 1
            physics_questions_raw = self.generate_physics_questions(prompt_text)
            more = self._extract_questions_from_generated(physics_questions_raw)
            for q in more:
                k = q.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(q.strip())
            # If still insufficient, continue looping until reaching target or retry limit
        # Truncate to 10
        if len(uniq) > target:
            uniq = uniq[:target]
        return uniq

    def evaluate_physics_reasoning(self, prompt_dict_ls: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
        """
        Evaluate physics reasoning ability

        Args:
            prompt_dict_ls: List of input data

        Returns:
            Tuple[float, List[Dict]]: (average score, detailed results)
        """
        total_score = 0.0
        valid_items = 0
        detailed_results = []

        for prompt_dict in tqdm(prompt_dict_ls):
            # Get video information
            print(f"===== Evaluating video list: {prompt_dict} =====")
            video_list = prompt_dict.get("video_list", [])
            prompt_text = prompt_dict.get("prompt", "")
            auxiliary_info = prompt_dict.get("auxiliary_info", [])

            for video_path in video_list:
                print(f"\n===== Evaluating video: {video_path} =====")

                # Check if there are pre-generated questions
                if auxiliary_info:
                    # Use pre-generated questions and ensure 10 questions (generate more if insufficient, truncate if excess)
                    print(f"Using {len(auxiliary_info)} pre-generated physics reasoning questions")
                    questions_init = [q for q in auxiliary_info if isinstance(q, str)]
                    questions = self._ensure_ten_questions(prompt_text, questions_init)
                else:
                    # Dynamically generate questions (one generation + additional if insufficient), ensure 10 questions
                    print("Dynamically generating physics reasoning questions...")
                    physics_questions_raw = self.generate_physics_questions(prompt_text)
                    questions_first = self._extract_questions_from_generated(physics_questions_raw)
                    questions = self._ensure_ten_questions(prompt_text, questions_first)

                if not questions:
                    print("Failed to generate valid physics questions, skipping this video")
                    continue

                print(f"Number of physics reasoning questions determined for evaluation: {len(questions)} (target: 10)")

                # Initialize results
                video_detail = {
                    "video_path": video_path,
                    "prompt": prompt_text,
                    "questions": questions,
                    "answers": [],
                    "question_details": [],
                }

                # Get video description
                print("Getting video description...")
                self.caption = Keye_describe_video(video_path, device=self.device)
                print(f"Video description: {self.caption}")

                # Answer questions concurrently (maintain order)
                total_questions = len(questions)
                answers = [None] * total_questions
                details = [None] * total_questions
                print(
                    f"Starting to answer {total_questions} questions concurrently, max concurrency: {self.max_workers}"
                )
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self.answer_physics_question, q, self.caption): idx
                        for idx, q in enumerate(questions)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            answer = future.result()
                        except Exception as e:
                            print(f"Concurrent answer exception idx={idx}: {e}")
                            answer = "null"
                        answers[idx] = answer
                        details[idx] = {
                            "question": questions[idx],
                            "video_caption": self.caption,
                            "answer": answer,
                            "is_correct": answer == "yes",
                        }

                # Write back and score
                video_detail["answers"] = answers
                video_detail["question_details"] = details

                correct_answers = sum(1 for a in answers if a == "yes")
                total_questions = len(questions)
                if total_questions > 0:
                    video_score = correct_answers / total_questions
                    video_detail["score"] = video_score
                    video_detail["correct_answers"] = correct_answers
                    video_detail["total_questions"] = total_questions

                    total_score += video_score
                    valid_items += 1

                    print("\nVideo evaluation completed:")
                    print(f"  - Correct answers: {correct_answers}/{total_questions}")
                    print(f"  - Score: {video_score:.4f}")
                else:
                    video_detail["score"] = 0.0
                    video_detail["correct_answers"] = 0
                    video_detail["total_questions"] = 0

                detailed_results.append(video_detail)

        # Calculate average score
        average_score = total_score / valid_items if valid_items > 0 else 0.0

        print("\n===== Physics Reasoning Evaluation Summary =====")
        print(f"Total videos: {valid_items}")
        print(f"Average score: {average_score:.4f}")
        print("================================================")

        return average_score, detailed_results


def compute_physics_realism(
    json_dir: str, device: str, submodules_dict: dict[str, Any], **kwargs
) -> tuple[float, list[dict[str, Any]]]:
    """
    Calculate physics reasoning score

    Args:
        json_dir: JSON file path
        submodules_dict: Submodules dictionary

    Returns:
        Tuple[float, List[Dict]]: (score, detailed results)
    """
    # Load data
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension="physics_realism", lang="en")

    # Resolve relative video paths using dataset base directory
    dataset_json = kwargs.get("dataset_json", "")
    dataset_base_dir = os.environ.get("DATASET_BASE_DIR", "")
    if not dataset_base_dir and dataset_json:
        # Infer base dir: dataset_json is like /base/condition_to_4D/.../file.json
        # and video paths start with condition_to_4D/...
        parts = dataset_json.split("/condition_to_4D/")
        if len(parts) > 1:
            dataset_base_dir = parts[0]
    if dataset_base_dir:
        for item in prompt_dict_ls:
            resolved = []
            for vp in item.get("video_list", []):
                if not os.path.isabs(vp) and not os.path.exists(vp):
                    full = os.path.join(dataset_base_dir, vp)
                    resolved.append(full)
                else:
                    resolved.append(vp)
            item["video_list"] = resolved

    # Ensure compatibility
    # if COMPATIBILITY_AVAILABLE:
    #     prompt_dict_ls = ensure_auxiliary_info_compatibility(prompt_dict_ls, 'physics_reasoning')

    # Create evaluator
    evaluator = PhysicsReasoningEvaluator(device)

    print("Starting physics reasoning evaluation...")
    # breakpoint()

    # Execute evaluation
    average_score, detailed_results = evaluator.evaluate_physics_reasoning(prompt_dict_ls)

    # Save detailed results
    try:
        output_dir = os.path.dirname(json_dir)
        dim_name = os.path.splitext(os.path.basename(json_dir))[0]
        model = kwargs.get("model", "")
        dataset_json = kwargs.get("dataset_json", "")
        dataset_base = os.path.splitext(os.path.basename(dataset_json))[0] if dataset_json else "dataset"
        suffix = f"{dim_name}__{model}__{dataset_base}_results.json" if model else f"{dim_name}_results.json"
        output_file = os.path.join(output_dir, suffix)

        detailed_output = {
            "evaluation_summary": {
                "total_videos": len(detailed_results),
                "total_score": round(sum([r.get("score", 0) for r in detailed_results]), 1),
                "average_score": average_score,
            },
            "video_details": detailed_results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        print(f"Error saving JSON file: {e!s}")

    return average_score, detailed_results


# if __name__ == "__main__":
#     # Test case
#     test_json_dir = "dimension_description_json/physics_reasoning.json"
#     device = "cpu"
#     submodules_dict = {}

#     score, results = compute_physics_reasoning(test_json_dir, device, submodules_dict)
#     print(f"Test completed, score: {score:.4f}")
