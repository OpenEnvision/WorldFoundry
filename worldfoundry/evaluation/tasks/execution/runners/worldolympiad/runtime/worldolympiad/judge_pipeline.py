'''
Video Physics LLM Judge Pipeline

基于 PhyGenBench 的提示词工程实践，实现单次调用的视频物理评估系统。

核心设计原则：
1. 明确的角色定义和任务描述
2. 结构化 JSON 输出
3. Chain-of-Thought 推理
4. 置信度量化
5. 多维度物理评估

python ./judge_pipeline.py path/to/video.mp4

'''

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import base64
from dotenv import load_dotenv

from model.openrouter import video_openrouter_call
from problem.problem_set import (
    ALL_QUESTIONS,
    MECHANICS_QUESTIONS,
    THERMOTICS_QUESTIONS,
    MATERIAL_QUESTIONS,
    PhysicsQuestion,
)

load_dotenv()


# ============================================================================
# 核心提示词设计：参考 PhyGenBench 的层次化评估方法
# ============================================================================

SYSTEM_PROMPT = """You are PhysicsJudge, a precise video physics evaluator with deep expertise in physical laws and visual analysis.

Your task is to analyze AI-generated videos and evaluate their adherence to real-world physics. You will be given ONE specific physics question to answer about the video.

CRITICAL INSTRUCTIONS:
1. Watch the video carefully and analyze the specific physical phenomenon in question
2. Think step-by-step about what you observe
3. Answer ONLY based on what actually happens in the video, not what should theoretically happen
4. If the phenomenon is not present, not visible, or unclear in the video, answer "NA"
5. Provide a confidence score based on visual clarity and certainty of your observation
6. Output STRICT JSON format - no markdown code blocks, no extra text

OUTPUT FORMAT:
Your response must be a valid JSON object with this exact structure:
{
  "answer": "yes|no|NA",
  "confidence": 0.0-1.0,
  "explanation": "Brief step-by-step reasoning for your answer"
}

EVALUATION PRINCIPLES:
- "yes": The physical phenomenon occurs correctly as expected by real-world physics
- "no": The physical phenomenon violates physics or occurs incorrectly
- "NA": The phenomenon is not present, not visible, or unclear in the video
- Confidence scale:
  * 1.0 = Absolutely certain, phenomenon is crystal clear
  * 0.7-0.9 = High confidence, phenomenon is clearly visible
  * 0.4-0.6 = Moderate confidence, some uncertainty due to video quality or ambiguity
  * 0.1-0.3 = Low confidence, phenomenon is barely visible or unclear
  * 0.0 = Cannot determine, pure guess
- Be conservative: Don't assume physics is correct unless the phenomenon is clearly visible and verifiable
"""


USER_PROMPT_TEMPLATE = """Analyze this video to answer the following physics question:

QUESTION: {question}

SUCCESS CONDITION: {success_condition}

ANALYSIS STEPS:
1. Watch the entire video carefully
2. Identify if the phenomenon described in the question occurs in the video
3. Evaluate whether the phenomenon follows correct physics (if it occurs)
4. Determine your answer: "yes" (physics is correct), "no" (physics is violated), or "NA" (not visible/not present)
5. Rate your confidence based on how clearly you can observe and evaluate the phenomenon
6. Provide a brief explanation of your reasoning

IMPORTANT REMINDERS:
- Focus on what you ACTUALLY SEE in the video, not what should happen theoretically
- If you cannot clearly see the phenomenon, use "NA" rather than guessing
- Think step-by-step before answering
- Consider whether the physics is CORRECT, not just whether something happens

Output your response in the required JSON format (no markdown, just pure JSON)."""


# ============================================================================
# Judge Pipeline Implementation
# ============================================================================

@dataclass
class JudgeResult:
    """Single question evaluation result"""
    qid: str
    dimension: str
    question: str
    answer: str  # "yes" | "no" | "NA"
    confidence: float
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "qid": self.qid,
            "dimension": self.dimension,
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class VideoJudgeResult:
    """Complete video evaluation result"""
    answers: List[JudgeResult]
    overall_physics_score: float
    overall_assessment: str
    raw_response: Dict

    def to_dict(self) -> Dict:
        return {
            "answers": [a.to_dict() for a in self.answers],
            "overall_physics_score": self.overall_physics_score,
            "overall_assessment": self.overall_assessment,
            "raw_response": self.raw_response,
        }

    def get_dimension_score(self, dimension: str) -> Optional[float]:
        """Calculate average score for a specific dimension"""
        relevant = [a for a in self.answers if a.dimension == dimension]
        if not relevant:
            return None

        # Score: yes=1.0, no=0.0, NA=0.5 (neutral), weighted by confidence
        scores = []
        for ans in relevant:
            if ans.answer.lower() == "yes":
                scores.append(ans.confidence * 1.0)
            elif ans.answer.lower() == "no":
                scores.append(ans.confidence * 0.0)
            else:  # NA
                scores.append(0.5)

        return sum(scores) / len(scores) if scores else None


def format_single_question(question: PhysicsQuestion) -> Dict[str, str]:
    """Format a single question for the prompt"""
    return {
        "question": question.question,
        "success_condition": question.success_condition,
    }


def parse_llm_response(response_text: str) -> Dict:
    """
    Parse LLM response with robust error handling
    Based on PhyGenBench's JSON cleaning strategy
    """
    # Remove markdown code blocks
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # If parsing fails, return error structure
        return {
            "error": "JSON parse failed",
            "raw_text": response_text,
            "exception": str(e)
        }


def judge_single_question(
    video_url: str,
    question: PhysicsQuestion,
    system_prompt: Optional[str] = None,
) -> JudgeResult:
    """
    Single API call to judge one physics question about a video

    Args:
        video_url: URL to the video file (data URL with base64 encoding)
        question: Single physics question to evaluate
        system_prompt: Custom system prompt (defaults to SYSTEM_PROMPT)

    Returns:
        JudgeResult with the evaluation for this single question
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    # Format the user prompt with this specific question
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question.question,
        success_condition=question.success_condition,
    )

    # Call OpenRouter API (Gemini 2.5 Flash) - ONE call per question
    response = video_openrouter_call(
        data_url=video_url,
        system_prompt=system_prompt,
        user_content=user_prompt,
        model_name="google/gemini-2.5-flash",
    )

    # Parse response
    if "choices" in response and len(response["choices"]) > 0:
        content = response["choices"][0]["message"]["content"]
        parsed = parse_llm_response(content)
    else:
        parsed = {"error": "No choices in response", "raw_response": response}

    print(f"Debug: LLM response for QID {question.qid}: {parsed}")

    # Handle parsing errors
    if "error" in parsed:
        return JudgeResult(
            qid=question.qid,
            dimension=question.dimension,
            question=question.question,
            answer="NA",
            confidence=0.0,
            explanation=f"Error: {parsed.get('error', 'Unknown error')}",
        )

    # Extract the result for this single question
    return JudgeResult(
        qid=question.qid,
        dimension=question.dimension,
        question=question.question,
        answer=parsed.get("answer", "NA"),
        confidence=float(parsed.get("confidence", 0.0)),
        explanation=parsed.get("explanation", ""),
    )


def judge_video_physics(
    video_url: str,
    questions: Optional[List[PhysicsQuestion]] = None,
    system_prompt: Optional[str] = None,
) -> VideoJudgeResult:
    """
    Judge multiple physics questions about a video (one API call per question)

    Args:
        video_url: URL to the video file (data URL with base64 encoding)
        questions: List of physics questions (defaults to ALL_QUESTIONS)
        system_prompt: Custom system prompt (defaults to SYSTEM_PROMPT)

    Returns:
        VideoJudgeResult with structured evaluation for all questions
    """
    if questions is None:
        questions = ALL_QUESTIONS

    # Evaluate each question with a separate API call
    answers = []
    for question in questions:
        result = judge_single_question(video_url, question, system_prompt)
        answers.append(result)

    # Calculate overall physics score
    # Score: yes with confidence, no = 0, NA = neutral 0.5
    scores = []
    for ans in answers:
        if ans.answer.lower() == "yes":
            scores.append(ans.confidence * 1.0)
        elif ans.answer.lower() == "no":
            scores.append(ans.confidence * 0.0)
        else:  # NA
            scores.append(0.5)

    overall_score = sum(scores) / len(scores) if scores else 0.0

    # Generate overall assessment
    yes_count = sum(1 for a in answers if a.answer.lower() == "yes")
    no_count = sum(1 for a in answers if a.answer.lower() == "no")
    na_count = sum(1 for a in answers if a.answer.lower() == "na")

    overall_assessment = (
        f"Evaluated {len(answers)} physics questions: "
        f"{yes_count} correct, {no_count} violations, {na_count} not applicable"
    )

    return VideoJudgeResult(
        answers=answers,
        overall_physics_score=overall_score,
        overall_assessment=overall_assessment,
        raw_response={"answers": [a.to_dict() for a in answers]},
    )


# ============================================================================
# Specialized Judge Functions
# ============================================================================

def judge_mechanics(video_url: str) -> VideoJudgeResult:
    """Evaluate only mechanics questions"""
    return judge_video_physics(video_url, questions=MECHANICS_QUESTIONS)


def judge_thermotics(video_url: str) -> VideoJudgeResult:
    """Evaluate only thermotics questions"""
    return judge_video_physics(video_url, questions=THERMOTICS_QUESTIONS)


def judge_materials(video_url: str) -> VideoJudgeResult:
    """Evaluate only material questions"""
    return judge_video_physics(video_url, questions=MATERIAL_QUESTIONS)


# ============================================================================
# Utility Functions
# ============================================================================

def print_judge_report(result: VideoJudgeResult, verbose: bool = True):
    """Pretty print judge results"""
    print("=" * 80)
    print("VIDEO PHYSICS EVALUATION REPORT")
    print("=" * 80)

    print(f"\nOVERALL PHYSICS SCORE: {result.overall_physics_score:.2f}")
    print(f"OVERALL ASSESSMENT: {result.overall_assessment}")

    # Group by dimension
    for dimension in ["mechanics", "thermotics", "material"]:
        dim_answers = [a for a in result.answers if a.dimension == dimension]
        if not dim_answers:
            continue

        dim_score = result.get_dimension_score(dimension)
        print(f"\n{dimension.upper()} (Score: {(f'{dim_score:.2f}' if dim_score else 'N/A')})")
        print("-" * 80)

        for ans in dim_answers:
            status_symbol = "✓" if ans.answer.lower() == "yes" else "✗" if ans.answer.lower() == "no" else "?"
            print(f"{status_symbol} [{ans.qid}] {ans.answer.upper()} (conf: {ans.confidence:.2f})")
            if verbose:
                print(f"  Question: {ans.question}")
                print(f"  Explanation: {ans.explanation}")
                print()

    print("=" * 80)


def save_judge_result(result: VideoJudgeResult, output_path: str):
    """Save result to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


# ============================================================================
# Example Usage
# ============================================================================


def get_frames_from_video(video_path: str, num_frames: int=8) -> List[str]:
    """
    Placeholder function to extract frames from video.
    In a real implementation, this would extract frames and return their URLs or base64 data.
    """
    import cv2
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames
    stride = max(total_frames // num_frames, 1)
    encoded_frames: List[str] = []

    frame_id = 0
    collected = 0
    while collected < num_frames and frame_id < total_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            break

        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
            encoded_frames.append(encoded)
            collected += 1
        frame_id += stride

    capture.release()
    return encoded_frames

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Physics LLM Judge")
    parser.add_argument("video_url", help="Path or URL to video file")
    parser.add_argument("--dimension", choices=["all", "mechanics", "thermotics", "material"],
                        default="all", help="Physics dimension to evaluate")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    use_video = True

    if use_video:
        def encode_video_to_base64(video_path):
            with open(video_path, "rb") as video_file:
                return base64.b64encode(video_file.read()).decode('utf-8')

        base64_video = encode_video_to_base64(args.video_url)

        data_url = f"data:video/mp4;base64,{base64_video}"
        data_url = {
            "type": "video_url",
            "video_url": {
                "url": data_url
            }
        }
    else:
        # get the frames from the video
        data_url = args.video_url
        data_url = get_frames_from_video(data_url, num_frames=8)
        target = []
        for frame in data_url:
            frame = f"data:image/jpg;base64,{frame}"
            target.append({
                "type": "image_url",
                "image_url": {
                    "url": frame
                }
            })
        data_url = target


    # Select evaluation function
    if args.dimension == "mechanics":
        result = judge_mechanics(data_url)
    elif args.dimension == "thermotics":
        result = judge_thermotics(data_url)
    elif args.dimension == "material":
        result = judge_materials(data_url)
    else:
        result = judge_video_physics(data_url)

    # Print report
    print_judge_report(result, verbose=not args.quiet)

    # Save if requested
    if args.output:
        save_judge_result(result, args.output)
