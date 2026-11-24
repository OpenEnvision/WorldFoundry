# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

Tool = dict[str, str | dict]
History = list[tuple[str, str] | list[str]]
Message = dict[str, str | list[dict[str, Any]]]
Messages = list[Message]


def messages_to_history(messages: Messages) -> dict[str, Any]:
    system = None
    messages = messages.copy()
    if messages[0]["role"] == "system":
        system = messages[0]["content"]
        messages = messages[1::]
    if len(messages) % 2 == 1:
        messages.append({"role": "assistant", "content": None})
    history = []
    history_roles = []
    for user_message, assistant_message in zip(messages[::2], messages[1::2], strict=False):
        assert user_message["role"] in {"tool", "user"}, f"user_message {user_message}"
        assert assistant_message["role"] == "assistant", f"assistant_message: {assistant_message}"
        history.append([user_message["content"], assistant_message["content"]])
        history_roles.append([user_message["role"], assistant_message["role"]])
    query, response = history.pop() if history else (None, None)
    query_role = history_roles.pop()[0] if history_roles else None
    return {
        "history": history,
        "history_roles": history_roles,
        "query": query,
        "query_role": query_role,
        "response": response,
        "system": system,
    }


@dataclass
class InferRequest:
    """
    messages: Input in messages format.
        Examples: [{
            "role": "user",  # or assistant/system/role
            "content": [  # str or List[Dict[str, Any]]
                {
                    "type": "image",  # or audio/video
                    "image": "<url/path/base64/PIL.Image>",
                },
                {"type": "text", "text": "Please describe the picture."},
            ],
        }]
        The above content is equivalent to:
        [{"role": "user", "content": "<image>Please describe the picture."}]
        and additionally passing in images: ["<url/path/base64/PIL.Image>"].
    tools: Organize tools into the format of agent_template for system. for example, 'react_en'.
    """

    messages: Messages

    images: list[str | Image.Image] = field(default_factory=list)  # noqa: F821 - Postponed PIL annotation retained from the template schema.
    audios: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)

    tools: list[Tool] | None = None
    objects: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self):
        for key in ["images", "audios", "videos"]:
            val = getattr(self, key)
            if isinstance(val, str):
                setattr(self, key, [val])
        assert isinstance(self.messages, list), f"messages: {self.messages}"

    @staticmethod
    def remove_response(messages) -> str | None:
        last_role = messages[-1]["role"] if messages else None
        if last_role == "assistant":
            return messages.pop()["content"]

    @staticmethod
    def _to_printable(obj, key: str | None = None):
        if isinstance(obj, str) and key not in {"content", "text"} and len(obj) >= 1000:
            return f"<<<base64:{obj[:50]}..>>>"
        elif isinstance(obj, list):
            res = []
            for item in obj:
                res.append(InferRequest._to_printable(item))
            return res
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = InferRequest._to_printable(v, key=k)
            return res
        return obj

    def to_printable(self):
        return InferRequest._to_printable(asdict(self))


@dataclass
class RolloutInferRequest(InferRequest):
    """
    A request class that modifies the 'images' attribute
    to be a list of strings for compatibility with POST requests.
    The strings can represent image URLs or Base64 encoded images.
    """

    images: list[str] = field(default_factory=list)


@dataclass
class TemplateInputs(InferRequest):
    """The training functionality has been added on top of the InferRequest.

    objects: Used for grounding tasks in a general format.
    """

    rejected_response: str | None = None
    label: bool | None = None


@dataclass
class StdTemplateInputs:
    # only user/tool/assistant
    messages: list[dict[str, str]]
    # None: use default system; '': not use system
    system: str | None = None
    tools: list[Tool] | None = None

    rejected_response: str | None = None
    label: int | None = None

    images: list[str | Image.Image] = field(default_factory=list)  # noqa: F821 - Postponed PIL annotation retained from the template schema.
    audios: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    objects: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self):
        self.image_idx = 0
        self.audio_idx = 0
        self.video_idx = 0
        self.ref_idx = 0
        self.bbox_idx = 0
        if self.images and not isinstance(self.images, (list, tuple)):
            self.images = [self.images]
        if self.videos and not isinstance(self.videos, (list, tuple)):
            self.videos = [self.videos]
        if self.audios and not isinstance(self.audios, (list, tuple)):
            self.audios = [self.audios]

    def to_history(self):
        if not self.messages:
            return None
        return messages_to_history(self.messages)

    @property
    def is_multimodal(self):
        return bool(self.images or self.audios or self.videos or self.objects)

    @classmethod
    def from_dict(cls, inputs: dict[str, Any]) -> StdTemplateInputs:
        kwargs = {}
        for key in ["rejected_response", "label"]:
            if key in inputs:
                kwargs[key] = inputs[key]
        messages = inputs["messages"]
        tools = inputs.get("tools")
        objects = inputs.get("objects") or {}

        if messages and messages[0]["role"] == "system":
            message = messages.pop(0)
            system = message["content"]
        else:
            system = None

        for message in messages:
            if message["role"] == "tool_response":
                message["role"] = "tool"
            if message["role"] in {"tool_call", "tool"} and not isinstance(message["content"], str):
                message["content"] = json.dumps(message["content"], ensure_ascii=False)

        media_kwargs = StdTemplateInputs.remove_messages_media(messages)
        for k in list(media_kwargs.keys()):
            mm_data = media_kwargs[k]

            inputs_mm_data = inputs.get(k)
            if isinstance(inputs_mm_data, str):
                inputs_mm_data = [inputs_mm_data]
            inputs_mm_data = (inputs_mm_data or []).copy()
            if mm_data:
                assert not inputs_mm_data, f"self.{k}: {inputs_mm_data}"
            else:
                media_kwargs[k] = inputs_mm_data

        return cls(messages=messages, system=system, tools=tools, objects=objects, **kwargs, **media_kwargs)

    @staticmethod
    def remove_messages_media(messages: Messages) -> dict[str, Any]:
        res = {"images": [], "audios": [], "videos": []}
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                continue
            # List[Dict[str, Any]]
            new_content = ""
            for item in content:
                key: str = item["type"]
                value = item.get(key)
                if key == "text":
                    new_content += value
                    continue
                # image/audio/video
                # image_url/audio_url/video_url
                if key.endswith("_url"):
                    key = key[: -len("_url")]
                new_content += f"<{key}>"
                if isinstance(value, dict):
                    value = value["url"]
                if value:
                    res[f"{key}s"].append(value)
            message["content"] = new_content
        return res
