"""Ask a VLM which physical phenomena a video involves."""

import base64
import json

from model.openrouter import video_openrouter_call

from problem.problem_set import MATERIAL_QUESTIONS, MECHANICS_QUESTIONS, THERMOTICS_QUESTIONS


def parse_llm_response(response_text: str) -> dict:
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
        return {"error": "JSON parse failed", "raw_text": response_text, "exception": str(e)}


def filter_problems_by_physics(video, model_name, video_prompt) -> dict:
    prompt = """
    You are a physics-aware problem filter. Given a Video Prompt, Video and the Problem Set, determine if the problem is relevant to the physical phenomena depicted in the video.

    output a JSON object with three keys: "mechanics", "thermotics", and "material". Each key maps to a list of question IDs from the Problem Set that are relevant to the video content.

    like
    {
        "mechanics": [gravity, buoyancy],
        "thermotics": [],
        "material": []
    }
    
    """

    prompt += f"\nVideo Prompt: {video_prompt}"

    prompt += f"\nProblems Set:{MECHANICS_QUESTIONS!s}, {THERMOTICS_QUESTIONS!s}, {MATERIAL_QUESTIONS!s}"

    print(prompt)

    SYSTEM_PROMPT = ""

    result = video_openrouter_call(
        data_url=video,
        user_content=prompt,
        system_prompt=SYSTEM_PROMPT,
        model_name=model_name,
    )

    print("Filter response:", result)

    parsed_response = parse_llm_response(result["choices"][0]["message"]["content"])

    return parsed_response


if __name__ == "__main__":
    video_path = "data/videos/impact_self_forcing_1.mp4"
    model_name = "google/gemini-2.5-flash"
    video_prompt = "A white cue ball striking a neatly arranged triangle of billiard balls on a green felt table. Upon impact, the triangle scatters in all directions, with balls losing momentum as they hit the cushions."

    def encode_video_to_base64(video_path):
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")

    base64_video = encode_video_to_base64(video_path=video_path)

    data_url = f"data:video/mp4;base64,{base64_video}"
    data_url = {"type": "video_url", "video_url": {"url": data_url}}

    result = filter_problems_by_physics(video_path, model_name, video_prompt)

    print("Filtered Problems:", result)
