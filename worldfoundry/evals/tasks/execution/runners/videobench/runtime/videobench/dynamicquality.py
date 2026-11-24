import logging
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .utils import Video_Dataset


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_api(client, messages, model):
    """Return the chat completion text for ``messages``; retried up to six times with random exponential backoff."""
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content


def extract_scores_from_result(response):
    because_matches = re.finditer(r"because", response)
    scores = []

    for match in because_matches:
        # The score is the last number before "because".
        preceding_numbers = re.findall(r"\d+", response[: match.start()])
        if preceding_numbers:
            scores.append(int(preceding_numbers[-1]))

    return scores


def eval(config, prompt, dimension, cur_full_info_path, models):
    """
    Evaluate videos using OpenAI API
    Args:
        config: configuration dictionary
        prompt: prompt template
        dimension: evaluation dimension name
    Returns:
        dict: containing evaluation scores
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(config[f"log_path_{dimension}"])
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    client = OpenAI(api_key=config["GPT4o_API_KEY"], base_url=config["GPT4o_BASE_URL"])
    MODEL = "gpt-4o-2024-08-06"

    results = {}
    dataset = Video_Dataset(cur_full_info_path)

    l1 = list(range(len(dataset)))
    model_scores = {model: {"total_score": 0, "count": 0} for model in models}

    for i in l1:
        try:
            data = dataset[i]
            frames = data["frames"]
            prompten = data["prompt"]

            results[i] = {}
            results[i]["prompt_en"] = data["prompt"]

            model_names = list(frames.keys())
            # One message with the frames of every model.
            model_frames_content = []
            for model_name, model_frames in frames.items():
                model_frames_content.extend(
                    [
                        f"\n{len(model_frames)} frames from {model_name}\n",
                        *(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpg;base64,{x}", "detail": "low"},
                            }
                            for x in model_frames
                        ),
                    ]
                )

            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        f"These are the frames from the video. The prompt is '{prompten}'.",
                        *model_frames_content,
                    ],
                },
            ]

            response = call_api(client, messages, MODEL)
            logger.info(f">>>>>>>This is the {i} round >>>>>>evaluation results>>>>>>:\n{response}\n")

            scores = extract_scores_from_result(response)

            # Scores appear in the same order as the models in the message.
            model_scores_dict = dict(zip(model_names, scores, strict=False))
            filtered_scores = {name: model_scores_dict[name] for name in models if name in model_scores_dict}
            results[i].update(filtered_scores)

            for model_name, score in filtered_scores.items():
                model_scores[model_name]["total_score"] += score
                model_scores[model_name]["count"] += 1

        except Exception as e:
            logger.error(f"Error processing video {i}: {e!s}")
            results[str(i)] = f"Error: {e!s}"

    average_scores = {
        model: model_scores[model]["total_score"] / model_scores[model]["count"]
        for model in model_scores
        if model_scores[model]["count"] > 0
    }

    return {
        "score": results,
        "average_scores": average_scores,  # mean score per model
    }
