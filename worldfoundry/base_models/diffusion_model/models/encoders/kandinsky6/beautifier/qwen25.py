# ruff: noqa: E501, W291
"""Qwen 2.5 beautifier. Reuses the text encoder's Qwen, no extra weights.

The instruction strings keep the original wording, including lines longer than
the project limit and one trailing space. The Russian speech lines are in English.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import transformers
from PIL import Image

ExpandMode = Literal["t2v", "t2va", "i2va"]


def _t2v_instruction(prompt: str) -> str:
    # Exact text from kandinsky-5-inference/kandinsky/t2v_pipeline.py::expand_prompt
    return f"""You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
        Here are some example descriptions from the dataset that the model was trained:
        1. "In a dimly lit room with a cluttered background, papers are pinned to the wall and various objects rest on a desk. Three men stand present: one wearing a red sweater, another in a black sweater, and the third in a gray shirt. The man in the gray shirt speaks and makes hand gestures, while the other two men look forward. The camera remains stationary, focusing on the three men throughout the sequence. A gritty and realistic visual style prevails, marked by a greenish tint that contributes to a moody atmosphere. Low lighting casts shadows, enhancing the tense mood of the scene."
        2. "In an office setting, a man sits at a desk wearing a gray sweater and seated in a black office chair. A wooden cabinet with framed pictures stands beside him, alongside a small plant and a lit desk lamp. Engaged in a conversation, he makes various hand gestures to emphasize his points. His hands move in different positions, indicating different ideas or points. The camera remains stationary, focusing on the man throughout. Warm lighting creates a cozy atmosphere. The man appears to be explaining something. The overall visual style is professional and polished, suitable for a business or educational context."
        3. "A person works on a wooden object resembling a sunburst pattern, holding it in their left hand while using their right hand to insert a thin wire into the gaps between the wooden pieces. The background features a natural outdoor setting with greenery and a tree trunk visible. The camera stays focused on the hands and the wooden object throughout, capturing the detailed process of assembling the wooden structure. The person carefully threads the wire through the gaps, ensuring the wooden pieces are securely fastened together. The scene unfolds with a naturalistic and instructional style, emphasizing the craftsmanship and the methodical steps taken to complete the task."
        IImportantly! These are just examples from a large training dataset of 200 million videos.
        Rewrite Prompt: "{prompt}" to get high-quality video generation. Answer only with expanded prompt."""


def _t2va_instruction(prompt: str) -> str:
    # K5 t2va instruction. The two spoken lines are translated from Russian.
    return f"""You are a prompt beautifier that transforms short user video+audio descriptions into rich, detailed English prompt specifically optimized for video+audio generation models.
    Here are some example descriptions from the dataset that the model was trained:
    1. "An extreme close-up of dark, out-of-focus tree branches fills the frame. The shallow depth of field renders them as soft, indistinct shapes against a blurred background of muted greens and grays. As the camera performs a slow pedestal-down movement, more of the surrounding foliage comes into view, revealing the dense forest setting. The focus remains intentionally soft throughout. The camera then executes another pedestal-down motion combined with a smooth tracking shot forward, bringing into clearer view a person partially obscured behind the trees. They are wearing a light-colored jacket or shirt and appear to be crouching low to the ground. Their head is slightly bowed, and their posture suggests they might be hiding or cautiously observing something off-screen. The ambient lighting is dim and diffused, consistent with being deep within a wooded area during overcast conditions or twilight hours. <AUDCAP>The gentle, melancholic melody of a solo piano plays softly in the background, its notes echoing faintly as if in a large, empty space. The music creates a somber and reflective mood that complements the mysterious atmosphere of the forest.<ENDAUDCAP>"
    2. "A woman with short brown hair styled into a neat updo sits facing slightly right of the camera. She wears a dark red sleeveless top, small dangling earrings featuring clear stones, and a thin silver necklace. Her makeup includes defined eyebrows, eyeliner, and neutral-toned lipstick. The background consists of out-of-focus wooden bookshelves filled with books and decorative items, suggesting an indoor setting like a study or library. The lighting is soft and warm, illuminating her face evenly from the front-left. Initially, she smiles gently as she speaks, her expression shifting subtly as she talks. Her head turns slightly toward the camera before returning to its previous angle. A man’s voice responds briefly after her initial statement. The shot remains static throughout, maintaining focus on her upper body and facial expressions. <S>I'm glad we've found common ground.<E> The woman speaks calmly and warmly, her voice carrying a gentle, inviting tone. <S>Maybe some tea, or coffee?<E> She repeats the phrase more slowly, her intonation light and suggestive, accompanied by a faint smile. <S>I'm glad we've found common ground.<E> Her voice returns, now softer and reflective, delivered with sincerity. <AUDCAP> Soft instrumental music plays in the background, creating a cozy ambiance. Gentle piano notes are audible beneath the dialogue, adding warmth to the intimate exchange.<ENDAUDCAP>"
    3. "A paraglider soars through the air above a vast landscape of rolling hills and forests during golden hour. The pilot, wearing a black jacket, tan pants, and a red helmet with goggles pushed up, sits securely within a blue circular harness suspended beneath a large, vibrant orange-red canopy. The canopy features subtle yellow and dark blue accents near its trailing edge and displays white lettering reading ""OZONE"" along its upper surface. The pilot holds control toggles in both hands and gestures upward with their right hand as they fly. Below, the terrain consists of brown fields, clusters of trees, scattered buildings including two long white-roofed structures, and a winding river visible in the distance under a pale sky streaked with soft clouds. <AUDCAP> A low, continuous rushing sound of wind fills the background, suggesting high altitude flight. There are no other discernible environmental noises or music.<ENDAUDCAP>""
    Importantly! These are just examples from a large training dataset of 20 mln videos.
    Rewrite Prompt: "{prompt}" to get high-quality text to video+audio generation. Make general audio part between <AUDCAP> <ENDAUDCAP>. Direct speech between <S> <E>.If initial prompt contains direct speech between <S> and <E>, then don't change it. If initial prompt contain request for speaking in some language without concrete direct speech,
    then add concrete direct speech between tags <S> and <E>.
    Make prompt dynamic. Answer only with expanded prompt."""


def _i2va_instruction(prompt: str) -> str:
    return f"""You are a prompt beautifier that transforms a short user video+audio description and a provided reference image into a rich, detailed English prompt optimized for video+audio generation.
    Rewrite Prompt: "{prompt}" to get high-quality image to video+audio generation.
    The reference image is the ground truth for the initial scene. Keep every visible fact that you mention consistent with it. Do not mainly describe the image: focus on the requested actions, motion, interactions, camera changes, speech, and audio, explaining the changes relative to the initial image. Add only details that are compatible with the reference image; never invent conflicting objects, identities, colors, locations, or actions.
    Preserve any direct speech between <S> and <E> exactly as written. If the prompt requests speech without exact words, add suitable direct speech between those tags. Put general audio descriptions between <AUDCAP> and <ENDAUDCAP>. Make the prompt dynamic and describe how the scene evolves from the provided image. Answer only with the expanded prompt."""


def _prepare_image(image: object) -> object:
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            return pil_image.convert("RGB").copy()
    return image


def build_expand_messages(
    prompt: str,
    mode: ExpandMode = "t2v",
    image: object | None = None,
) -> list[dict]:
    """Chat messages fed to Qwen for prompt expansion (testable without weights)."""
    if mode == "t2va":
        instruction = _t2va_instruction(prompt)
    elif mode == "i2va":
        instruction = _i2va_instruction(prompt)
    elif mode == "t2v":
        instruction = _t2v_instruction(prompt)
    else:
        raise ValueError(f"Unknown expand mode: {mode!r}")
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": instruction})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def expand_prompt(  # noqa: PLR0913
    prompt: str,
    qwen_processor,
    qwen_model,
    max_new_tokens: int = 256,
    mode: ExpandMode = "t2v",
    image: object | None = None,
) -> str:
    """Rewrite a short prompt into a detailed generation description using Qwen2.5-VL."""
    image = _prepare_image(image) if image is not None else None
    messages = build_expand_messages(prompt, mode=mode, image=image)
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(
        text=[text],
        images=[image] if image is not None else None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(qwen_model.device)
    # Newer processors emit this; Qwen2.5-VL.generate does not accept it.
    inputs.pop("mm_token_type_ids", None)

    generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    qwen_crop_start = inputs["input_ids"].shape[1]
    trimmed = [out[qwen_crop_start:] for out in generated_ids]
    return qwen_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class Qwen25Beautifier:
    name = "qwen25"

    def expand(
        self,
        prompt: str,
        *,
        image: object | None = None,
        audio: bool = False,
        seed: int | None = None,
        text_embedder=None,
    ) -> str:
        if text_embedder is None:
            raise RuntimeError("qwen25 beautifier needs the pipeline text embedder")
        if seed is not None:
            transformers.set_seed(seed)
        if image is not None:
            mode = "i2va"
        elif audio:
            mode = "t2va"
        else:
            mode = "t2v"
        return expand_prompt(
            prompt,
            text_embedder.processor,
            text_embedder.qwen,
            max_new_tokens=text_embedder.max_length,
            mode=mode,
            image=image,
        )
