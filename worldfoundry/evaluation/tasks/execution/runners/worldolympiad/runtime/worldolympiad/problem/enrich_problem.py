'''
Docstring for problem.problem_filter

这个文件提供一个接口，让VLM来提出具体的物理现象相关的问题，具体到物体和内容
'''

from problem.problem_set import MECHANICS_QUESTIONS, THERMOTICS_QUESTIONS, MATERIAL_QUESTIONS, get_question_by_id

from model.openrouter import video_openrouter_call
import base64
from typing import Dict


def enrich_problems(video, model_name, video_prompt, related_problem: Dict) -> str:
    prompt = f"""
    You are a physics-aware problem enricher. Given a Video Prompt, Video and the Problem Set, generate detailed physics problems related to the physical phenomena depicted in the video.

    output a JSON object with three keys: "mechanics", "thermotics", and "material". Each key maps to a list of enriched physics problems based on the Problem Set.

    like
    {{
        "mechanics": ["Problem 1 details...", "Problem 2 details..."],
        "thermotics": [],
        "material": []
    }}
    
    """

    for key, problem_ids in related_problem.items():
        if problem_ids:
            for pid in problem_ids:
                question = get_question_by_id(pid)
                if question:
                    prompt += f"\nRelated {key} Problems Details: {str(question)}"

    print(prompt)

    prompt += f"\nVideo Prompt: {video_prompt}"

    result = video_openrouter_call(
        data_url=video,
        system_prompt="You are a helpful physics problem generator.",
        user_content=prompt,
        model_name=model_name,
    )

    print("Enrich response:", result)

    return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    video_path = "runs/segment/predict/buoyancy_self_forcing_2.mp4"
    model_name = "google/gemini-2.5-flash"
    video_prompt = "A slice of fresh lemon is dropped into a glass of sparkling water. The lemon submerged initially but quickly floats back to the top, swaying slightly amongst the rising bubbles."
    related_problem = {'mechanics': ['gravity', 'buoyancy'], 'thermotics': [], 'material': []}
    def encode_video_to_base64(video_path):
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')

    base64_video = encode_video_to_base64(video_path=video_path)

    data_url = f"data:video/mp4;base64,{base64_video}"
    data_url = {
        "type": "video_url",
        "video_url": {
            "url": data_url
        }
    }

    result = enrich_problems(video_path, model_name, video_prompt, related_problem)

    print("Filtered Problems:", result)
