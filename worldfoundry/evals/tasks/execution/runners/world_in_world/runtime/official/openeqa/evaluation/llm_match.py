# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import traceback

from openeqa.utils.openai_utils import (
    call_openai_api,
    prepare_openai_messages,
    set_openai_key,
)
from openeqa.utils.prompt_utils import load_prompt


def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError(f"Invalid output string: {output}")
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())


def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: list | None = None,
    openai_key: str | None = None,
    openai_model: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    openai_base_url: str | None = None,
    verbose: bool = False,
    diff_retry_setting: bool = True,
):
    if prediction is None:
        return 0

    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

    retry_count = -1
    while retry_count < 5:
        try:
            retry_count += 1
            if diff_retry_setting:
                openai_max_tokens += 10
                openai_seed = random.randint(1, 100)
                openai_temperature += 0.2
            set_openai_key(key=openai_key)
            messages = prepare_openai_messages(
                prompt.format(
                    question=question,
                    answer=answer,
                    prediction=prediction,
                    extra_answers=extra_answers,
                ),
            )
            output = call_openai_api(
                messages=messages,
                model=openai_model,
                seed=openai_seed,
                max_tokens=openai_max_tokens,
                temperature=openai_temperature,
                verbose=verbose,
                base_url=openai_base_url,
            )
            return parse_score(output)
        except ValueError as e:
            openai_args = {
                "openai_max_tokens": openai_max_tokens,
                "openai_seed": openai_seed,
                "openai_temperature": openai_temperature,
            }
            print(f"[Retry {retry_count}] ValueError: {e}. Raw Response: {output}. {openai_args}")
        except Exception as e:
            traceback.print_exc()
            raise e
    # raise ValueError("Failed to get LLM match score")
    print("WARNING: Failed to get LLM match score, return 0")
    return 0


if __name__ == "__main__":
    # example usage
    question = "What color is the rug?"
    answer = "tan with pink and blue"
    prediction = "brown with pink and blue"
    score = get_llm_match_score(question, answer, prediction)
    print("*" * 40)
    print(f"example question:    {question}")
    print(f"ground-truth answer: {answer}")
    print(f"predicted answer:    {prediction}")
    print(f"llm-match score:     {score}")
    print("*" * 40)
