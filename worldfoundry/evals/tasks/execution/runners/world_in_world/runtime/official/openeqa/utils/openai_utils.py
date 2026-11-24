# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
import os

import cv2
import openai


def set_openai_key(key: str | None = None):
    if key is None:
        assert "OPENAI_API_KEY" in os.environ
        key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key


def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]


def prepare_openai_vision_messages(
    prefix: str | None = None,
    suffix: str | None = None,
    image_paths: list[str] | None = None,
    image_size: int | None = 512,
):
    if image_paths is None:
        image_paths = []

    content = []

    if prefix:
        content.append({"text": prefix, "type": "text"})

    for path in image_paths:
        frame = cv2.imread(path)
        if image_size:
            factor = image_size / max(frame.shape[:2])
            frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
        _, buffer = cv2.imencode(".png", frame)
        frame = base64.b64encode(buffer).decode("utf-8")
        content.append(
            {
                "image_url": {"url": f"data:image/png;base64,{frame}"},
                "type": "image_url",
            }
        )

    if suffix:
        content.append({"text": suffix, "type": "text"})

    return [{"role": "user", "content": content}]


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai_api(
    messages: list,
    model: str = "gpt-4",
    seed: int | None = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
    base_url: str | None = None,
):
    if base_url is None:
        client = openai.OpenAI(api_key=openai.api_key)
    else:
        client = openai.OpenAI(api_key=openai.api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print(f"openai api response: {completion}")
    assert len(completion.choices) == 1
    return completion.choices[0].message.content


if __name__ == "__main__":
    set_openai_key(key=None)

    messages = prepare_openai_messages("What color are apples?")
    print("input:", messages)

    model = "gpt-4-vision-preview"
    output = call_openai_api(messages, model=model, max_tokens=512, temperature=1.0)
    print(f"output: {output}")
