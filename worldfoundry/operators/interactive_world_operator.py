"""Shared request normalization for indexed and interactive inference routes."""

from collections.abc import Mapping

from .base_operator import BaseOperator


class InteractiveWorldOperator(BaseOperator):
    """Keep model controls explicit and strip evaluation bookkeeping once."""

    def __init__(self, input_schema=None):
        super().__init__(["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = dict(input_schema or {})

    def check_interaction(self, interaction):
        if interaction is not None and not isinstance(interaction, Mapping):
            raise TypeError("Pass named controls in a mapping; see the selected model's inference guide.")
        return True

    def get_interaction(self, interaction):
        self.check_interaction(interaction)
        self.current_interaction.append(dict(interaction or {}))

    def process_interaction(self):
        result = dict(self.current_interaction[-1]) if self.current_interaction else {}
        self.interaction_history.append(result)
        return result

    def process_perception(
        self, *, prompt=None, images=None, video=None, interactions=None, operator_kwargs=None, **kwargs
    ):
        if video is not None:
            raise ValueError(
                "This route does not accept input video; use its documented first-frame or indexed inputs."
            )
        self.check_interaction(interactions)
        request = {**dict(interactions or {}), **dict(operator_kwargs or {}), **kwargs}
        reference = request.pop("ref_image_path", None)
        if images is None and reference is not None:
            images = reference
        for key in ("task_name", "sample_id"):
            request.pop(key, None)
        if prompt is not None:
            request["prompt"] = prompt
        if images is not None:
            request["images"] = images
        return request
