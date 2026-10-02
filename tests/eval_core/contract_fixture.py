from __future__ import annotations

from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import ArtifactRef, GenerationRequest, GenerationResult, WorldModelConfig


CONTRACT_FIXTURE_MODEL_ID = "test-contract-model"
CONTRACT_FIXTURE_RUNNER_TARGET = "tests.eval_core.contract_fixture:ContractFixtureRunner"
CONTRACT_FIXTURE_RUNNER_ALIAS = "test:contract"


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


class ContractFixtureRunner:
    capabilities = {"worldfoundry.test.contract"}

    def __init__(
        self,
        model_id: str = CONTRACT_FIXTURE_MODEL_ID,
        *,
        output_artifacts: Sequence[str] = (),
        metrics: Mapping[str, Any] | None = None,
        artifact_uri_template: str = "memory://{sample_id}/{artifact_name}.json",
        metadata_namespace: str | None = None,
    ) -> None:
        self.model_id = str(model_id)
        self.output_artifacts = tuple(str(item) for item in output_artifacts)
        self.metrics = dict(metrics or {})
        self.artifact_uri_template = artifact_uri_template
        self.metadata_namespace = metadata_namespace
        self.cleaned = False

    @classmethod
    def from_config(cls, config: WorldModelConfig) -> "ContractFixtureRunner":
        parameters = dict(config.parameters or {})
        runtime = dict(config.runtime or {})
        output_artifacts = _str_tuple(
            parameters.get("output_artifacts")
            or runtime.get("output_artifacts")
            or (config.manifest.output_artifacts if config.manifest is not None else ())
            or parameters.get("artifact_kind")
        )
        metrics = parameters.get("metrics") if isinstance(parameters.get("metrics"), Mapping) else None
        artifact_ext = str(parameters.get("artifact_ext", "")).lstrip(".")
        artifact_uri_template = str(
            parameters.get(
                "artifact_uri_template",
                f"memory://{{sample_id}}.{{artifact_ext}}" if artifact_ext else "memory://{sample_id}/{artifact_name}.json",
            )
        )
        if artifact_ext:
            artifact_uri_template = artifact_uri_template.replace("{artifact_ext}", artifact_ext)
        return cls(
            model_id=config.model_id,
            output_artifacts=output_artifacts,
            metrics=metrics,
            artifact_uri_template=artifact_uri_template,
            metadata_namespace=parameters.get("metadata_namespace") or runtime.get("metadata_namespace"),
        )

    def _output_keys(self, request: GenerationRequest) -> tuple[str, ...]:
        if request.output_schema:
            return tuple(str(key) for key in request.output_schema)
        return self.output_artifacts or ("generated_artifact",)

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for request in requests:
            artifacts = {
                name: ArtifactRef(
                    uri=self.artifact_uri_template.format(
                        sample_id=request.sample_id,
                        request_id=request.request_id or request.sample_id,
                        artifact_name=name,
                    ),
                    kind=name,
                    metadata={"test_fixture": True},
                )
                for name in self._output_keys(request)
            }
            metrics = {"generation_success": 1.0, "task_success": 1.0, "success": 1.0, **self.metrics}
            metadata: dict[str, Any] = {
                "test_fixture": True,
                "metrics": metrics,
                "outputs": {"output_keys": list(artifacts)},
                "request_task_name": request.task_name,
            }
            if self.metadata_namespace:
                metadata[self.metadata_namespace] = {
                    "test_fixture": True,
                    "task_name": request.task_name,
                    "metrics": metrics,
                    "outputs": {"output_keys": list(artifacts)},
                }
            results.append(
                GenerationResult(
                    sample_id=request.sample_id,
                    request_id=request.request_id,
                    model_id=self.model_id,
                    artifacts=artifacts,
                    metadata=metadata,
                )
            )
        return results

    def cleanup(self) -> None:
        self.cleaned = True
