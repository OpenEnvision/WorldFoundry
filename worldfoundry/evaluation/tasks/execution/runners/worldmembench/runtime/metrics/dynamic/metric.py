# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Run VLM answer generation and score calculation through one evaluation interface."""

from dataclasses import asdict

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.aggregation import (
    aggregate_video_summary,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.base import Metric
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import (
    ItemResult,
    ResultStatus,
    UnitResult,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import VideoReader

from .checklist import parse_checklist
from .scoring import aggregate_scores, score_answers
from .vlm import HTTPVLMBackend, InvalidVLMResponse, VLMConfig, generate_answers


class DynamicMemoryMetric(Metric):
    name = "dynamic_memory"
    description = "VLM answers to eight visual questions, scored against case-specific targets."

    def __init__(self, *, backend=None, reader=None):
        self.backend = backend
        self.reader = reader or VideoReader()

    def evaluate(self, sample, context):
        if sample.get("category") not in (None, "dynamic_memory"):
            return [UnitResult("checklist", status=ResultStatus.NOT_APPLICABLE)]
        if error := sample.get("dynamic_preparation_error"):
            raise DataInvalidError(error["message"])
        config = VLMConfig(**context.metric_config(self.name))
        annotations = sample["annotations"]
        checklist = parse_checklist(annotations["dynamic_checklist"])
        backend = self.backend or context.get_resource("dynamic:vlm", HTTPVLMBackend)
        try:
            response, indices, protocol, anchors = generate_answers(
                sample["video"],
                checklist,
                annotations["lr_hold_protocol"],
                config,
                reader=self.reader,
                backend=backend,
            )
        except InvalidVLMResponse as error:
            return [
                UnitResult(
                    "checklist",
                    status=ResultStatus.EVAL_FAILED,
                    error=str(error),
                    metadata={"raw_vlm_response": error.raw_response},
                )
            ]
        units = score_answers(checklist, response, indices)
        event = response.event_started
        units[0].metadata["case_metadata"] = {
            "case_id": checklist.case_id,
            "vlm_model_id": config.model,
            "vlm_backend": config.backend,
            "vlm_settings": {key: value for key, value in asdict(config).items() if key != "api_key_env"},
            "sampled_frame_indices": indices,
            "event_started": event.answer,
            "event_gate_applied": event.answer == "no",
            "event_evidence_image_numbers": list(event.evidence_image_numbers),
            "event_evidence_frame_indices": [indices[n - 1] for n in event.evidence_image_numbers],
            "event_reason": event.reason,
            "raw_vlm_response": response.raw_response,
            "lr_hold_protocol": protocol.as_dict(),
            "prompt_anchors": anchors.as_dict(),
            "usage": response.usage,
        }
        return units

    def aggregate_item(self, sample, units):
        if units[0].status != ResultStatus.OK:
            unit = units[0]
            return ItemResult(sample["item_id"], status=unit.status, error=unit.error, metadata=dict(unit.metadata))
        scores = aggregate_scores(units)
        metadata = dict(units[0].metadata["case_metadata"])
        metadata["questions"] = [
            {
                "id": unit.unit_id,
                "score": unit.scores["score"],
                **{key: value for key, value in unit.metadata.items() if key != "case_metadata"},
            }
            for unit in units
        ]
        return ItemResult(sample["item_id"], scores=scores, num_units=8, metadata=metadata)

    def aggregate_summary(self, items):
        return aggregate_video_summary(items)
