import logging
import re

from openai import OpenAI

from .utils import Video_Dataset


def eval(config, prompt, dimension, cur_full_info_path, models):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(config[f"log_path_{dimension}"])
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    client = OpenAI(api_key=config["GPT4o_API_KEY"], base_url=config["GPT4o_BASE_URL"])

    MODEL = "gpt-4o-2024-08-06"

    dataset = Video_Dataset(cur_full_info_path)
    results = {}
    model_scores = {}

    l1 = list(range(len(dataset)))
    for i in l1:
        logger.info(f">>>>>>>>This is the {i} round>>>>>>>")
        data = dataset[i]
        frames = data["frames"]
        prompten = data["prompt"]
        results[i] = {}
        results[i]["prompt_en"] = prompten

        available_models = list(data["frames"].keys())

        for modelname in models:
            if modelname not in model_scores:
                model_scores[modelname] = {"total_score": 0, "count": 0}

            try:
                # Frames of the other models serve as references.
                examplemodels = [x for x in available_models if x not in models]

                messages = [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            "According to **Important Notes** in system meassage, there are examples from other models.\n",
                            *[
                                item
                                for examplemodel in examplemodels
                                for item in [
                                    f"This example is from model {examplemodels.index(examplemodel) + 1} \n",
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpg;base64,{frames[examplemodel][0]}",
                                            "detail": "low",
                                        },
                                    },
                                ]
                            ],
                            "These are the frames from the video you are evaluating. \n",
                            *(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpg;base64,{x}", "detail": "low"},
                                }
                                for x in frames[modelname]
                            ),
                            "Assuming there are a video ' scoring 'x',provide your analysis and explanation in the output format as follows:\n"
                            "- video: x ,because ...",
                        ],
                    },
                ]

                response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
                response_content = response.choices[0].message.content
                logger.info(f">>>>>>>This is the {i} round >>>>>>evaluation results>>>>>>:\n{response_content}\n")

                match = re.search(r":\s*(\d+)", response_content)
                video_score = int(match.group(1)) if match else "Error"
                results[i][modelname] = video_score

                model_scores[modelname]["total_score"] += video_score
                model_scores[modelname]["count"] += 1

            except Exception as e:
                logger.error(f"Error evaluating model {modelname}: {e!s}")
                results[i][modelname] = "Error"

    average_scores = {
        model: model_scores[model]["total_score"] / model_scores[model]["count"]
        for model in model_scores
        if model_scores[model]["count"] > 0
    }

    return {
        "score": results,
        "average_scores": average_scores,  # mean score per model
    }
