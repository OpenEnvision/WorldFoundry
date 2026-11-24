"""LoRA-based video instance anomaly inference used by VBench 2.0."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from .split import split

if TYPE_CHECKING:
    from worldfoundry.base_models.llm_mllm_core.mllm.modelscope_swift.swift.llm import (
        InferRequest,
        PtEngine,
        RequestConfig,
    )


def _infer_lora(engine: PtEngine, request_config: RequestConfig, infer_request: InferRequest) -> str:
    response = engine.infer([infer_request], request_config)[0].choices[0].message.content
    return response or ""


def _video_info(video_path: str) -> tuple[int, float, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 1.0)
    cap.release()
    return frame_count, fps, int(frame_count // max(fps, 1.0))


def _configure_video_limits() -> None:
    os.environ.setdefault("MAX_PIXELS", "1003520")
    os.environ.setdefault("VIDEO_MAX_PIXELS", "602112")
    os.environ.setdefault("VIDEO_MIN_PIXELS", "100352")
    os.environ.setdefault("FPS_MAX_FRAMES", "50")
    os.environ.setdefault("FPS_MIN_FRAMES", "32")


def _load_adapter_infer_args(adapter_path: str) -> SimpleNamespace:
    args_path = os.path.join(adapter_path, "args.json")
    with open(args_path, encoding="utf-8") as f:
        args = json.load(f)
    missing = [key for key in ("model", "template") if not args.get(key)]
    if missing:
        raise ValueError(f"Adapter args missing required inference keys {missing}: {args_path}")
    return SimpleNamespace(
        model=args["model"],
        template=args["template"],
        system=args.get("system"),
    )


def compute_anomaly(prompt_dict_ls: list[dict[str, Any]], device: str, submodules_dict: dict[str, str]):
    from tqdm import tqdm

    from worldfoundry.base_models.llm_mllm_core.mllm.modelscope_swift.swift.llm import (
        InferRequest,
        PtEngine,
        RequestConfig,
        get_model_tokenizer,
        get_template,
    )
    from worldfoundry.base_models.llm_mllm_core.mllm.modelscope_swift.swift.tuners import Swift

    _configure_video_limits()
    request_config = RequestConfig(max_tokens=512, temperature=0)
    adapter_path = submodules_dict["model"]
    args = _load_adapter_infer_args(adapter_path)
    model, tokenizer = get_model_tokenizer(args.model)
    model = Swift.from_pretrained(model, adapter_path)
    template = get_template(args.template, tokenizer, args.system)
    engine = PtEngine.from_model_template(model, template)

    final_score = 0
    final_num = 0
    processed_json = []
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict["video_list"]
        model_name = video_paths[0].split("/")[-3]
        split(os.path.dirname(video_paths[0]), model_name)
        video_clip_paths = f"./model_clip/{model_name}"
        for video_path in video_paths:
            _, fps, video_length = _video_info(video_path)
            os.environ["FPS"] = f"{fps}"
            valid = True
            new_item = {"video_path": video_path}
            for idx in range(max(video_length - 1, 0)):
                clip_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_clip_{idx:04d}.mp4"
                clip_path = os.path.join(video_clip_paths, clip_name)
                message = [
                    {"role": "system", "content": "You are a helpful and harmless assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": clip_path},
                            {
                                "type": "text",
                                "text": (
                                    "Does the video contain one or more of the following anomalies: "
                                    "sudden appearance, disappearance, fusion, fission?\n"
                                    "Options:\nA. Yes\nB. No"
                                ),
                            },
                        ],
                    },
                ]
                infer_request = InferRequest(messages=message)
                try:
                    output_text = _infer_lora(engine, request_config, infer_request)
                except Exception:
                    output_text = "no"
                    print(f"Error processing video: {clip_path}")
                if "(A)" in output_text or "yes" in output_text.lower() or output_text == "A":
                    valid = False
                    break
            new_item["video_results"] = 1.0 if valid else 0.0
            final_score += int(valid)
            final_num += 1
            processed_json.append(new_item)
    return final_score / max(final_num, 1), processed_json
