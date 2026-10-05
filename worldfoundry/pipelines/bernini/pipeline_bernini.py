"""WorldFoundry pipeline lifecycle adapter for all Bernini variants and tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from worldfoundry.core.io import artifact_root_path, copy_uri, is_remote_uri, local_path_for_uri, uri_to_local_path
from worldfoundry.core.utils import load_pil_image, materialize_image_input
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.bernini.worldfoundry_runtime import (
    MODEL_CONFIG_KEYS,
    BerniniRuntime,
    infer_task_type,
)

if TYPE_CHECKING:
    from worldfoundry.core.contracts import PipelineInvocation

GENERATION_KEYS = frozenset(
    {
        "neg_prompt",
        "num_frames",
        "max_image_size",
        "height",
        "width",
        "num_inference_steps",
        "guidance_mode",
        "omega_vid",
        "omega_img",
        "omega_txt",
        "omega_tgt",
        "omega_scale",
        "planning_step",
        "vit_txt_cfg",
        "vit_img_cfg",
        "vit_denoising_step",
        "flow_shift",
        "seed",
        "fps",
        "eta",
        "norm_threshold",
        "momentum",
        "system_prompt",
        "use_truncate",
        "max_sequence_length",
        "nproc_per_node",
        "ulysses_size",
        "plan_only",
    }
)


class BerniniWorldFoundryPipeline(PipelineABC):
    """Lazy native pipeline shared by Bernini and Bernini-R checkpoints."""

    def __init__(self, runtime: BerniniRuntime) -> None:
        self.runtime = runtime

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str = "bernini",
        **kwargs: Any,
    ) -> "BerniniWorldFoundryPipeline":
        options: dict[str, Any] = {}
        if isinstance(model_path, Mapping):
            options.update(model_path)
        elif model_path:
            options["checkpoint_path"] = model_path
        options.update(dict(required_components or {}))
        options.update(kwargs)
        checkpoint_path = next(
            (
                options[key]
                for key in ("checkpoint_path", "config_dir", "pretrained_model_path", "repo_root")
                if options.get(key)
            ),
            None,
        )
        model_config = {key: options[key] for key in MODEL_CONFIG_KEYS if key in options}
        return cls(
            BerniniRuntime(
                model_id=model_id,
                checkpoint_path=checkpoint_path,
                device=device,
                model_config=model_config,
            )
        )

    @staticmethod
    def _local_media(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [BerniniWorldFoundryPipeline._local_media(item) for item in value if item is not None]
        if isinstance(value, Path):
            return str(value.expanduser().resolve())
        if isinstance(value, str):
            if is_remote_uri(value):
                return value
            local = uri_to_local_path(value)
            return str(local.resolve()) if local.is_file() else value
        return value

    @staticmethod
    def _materialize_images(values: Sequence[Any], output_path: Path) -> list[str]:
        input_dir = output_path.parent / f".{output_path.stem}_inputs"
        paths: list[str] = []
        for index, value in enumerate(values):
            localized = BerniniWorldFoundryPipeline._local_media(value)
            if isinstance(localized, str) and Path(localized).is_file():
                paths.append(str(Path(localized).resolve()))
            elif isinstance(localized, str) and is_remote_uri(localized):
                with local_path_for_uri(localized) as local:
                    paths.append(
                        materialize_image_input(
                            local,
                            input_dir,
                            filename=f"reference_{index:02d}.png",
                        )
                    )
            else:
                paths.append(
                    materialize_image_input(
                        load_pil_image(localized),
                        input_dir,
                        filename=f"reference_{index:02d}.png",
                    )
                )
        return paths

    @staticmethod
    def _materialize_videos(value: Any, output_path: Path) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [BerniniWorldFoundryPipeline._materialize_videos(item, output_path) for item in value]
        localized = BerniniWorldFoundryPipeline._local_media(value)
        if not isinstance(localized, str) or not is_remote_uri(localized):
            return localized
        suffix = Path(localized.split("?", 1)[0]).suffix or ".mp4"
        target = output_path.parent / f".{output_path.stem}_inputs" / f"source_video{suffix}"
        copy_uri(localized, target)
        return str(target.resolve())

    @staticmethod
    def _first_present(*values: Any) -> Any:
        return next((value for value in values if value is not None), None)

    def run_pipeline_invocation(self, invocation: PipelineInvocation) -> Mapping[str, Any]:
        pipeline_kwargs = dict(invocation.pipeline_kwargs)
        operator_kwargs = dict(invocation.operator_kwargs)
        explicit_task = pipeline_kwargs.pop("task_type", None) or operator_kwargs.get("task_type")
        multi_images = self._first_present(
            operator_kwargs.get("images"),
            operator_kwargs.get("reference_images"),
            (),
        )
        image_value = self._first_present(
            invocation.image,
            operator_kwargs.get("conditioning_image"),
            operator_kwargs.get("first_frame"),
        )
        image_values = (
            list(multi_images)
            if isinstance(multi_images, (list, tuple))
            else ([] if multi_images is None else [multi_images])
        )
        if image_value is not None and not image_values:
            image_values = [image_value]
        task_type = infer_task_type(
            explicit=explicit_task,
            image=image_value,
            images=image_values,
            video=invocation.video,
            num_frames=pipeline_kwargs.get("num_frames"),
            output_schema=invocation.request.output_schema,
        )
        materialized_images = self._materialize_images(image_values, invocation.output_path) if image_values else []
        single_image = materialized_images[0] if task_type == "i2i" and materialized_images else None
        reference_images = materialized_images if task_type in {"r2v", "rv2v"} else None
        generation = {key: value for key, value in pipeline_kwargs.items() if key in GENERATION_KEYS}
        for key in GENERATION_KEYS:
            if key in operator_kwargs and key not in generation:
                generation[key] = operator_kwargs[key]
        return self.runtime.generate(
            prompt=invocation.prompt,
            output_path=invocation.output_path,
            task_type=task_type,
            image=single_image,
            images=reference_images,
            video=self._materialize_videos(invocation.video, invocation.output_path),
            **generation,
        )

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        target = (
            Path(output_path)
            if output_path
            else artifact_root_path() / "pipeline_eval" / f"{self.runtime.variant.model_id}.mp4"
        )
        images = self._first_present(
            images,
            kwargs.pop("image", None),
            kwargs.pop("conditioning_image", None),
            kwargs.pop("reference_images", None),
        )
        image_values = list(images) if isinstance(images, (list, tuple)) else ([images] if images is not None else [])
        task_type = infer_task_type(
            explicit=kwargs.pop("task_type", None),
            image=image_values[0] if len(image_values) == 1 else None,
            images=image_values,
            video=video,
            num_frames=kwargs.get("num_frames"),
        )
        materialized = self._materialize_images(image_values, target) if image_values else []
        result = self.runtime.generate(
            prompt=prompt,
            output_path=target,
            task_type=task_type,
            image=materialized[0] if task_type == "i2i" and materialized else None,
            images=materialized if task_type in {"r2v", "rv2v"} else None,
            video=self._materialize_videos(video, target),
            **{key: value for key, value in kwargs.items() if key in GENERATION_KEYS},
        )
        if return_dict:
            return result
        if result.get("status") != "succeeded":
            raise RuntimeError(result.get("blocked_reason") or result.get("error") or "Bernini inference failed")
        return result["artifact_path"]


__all__ = ["BerniniWorldFoundryPipeline", "GENERATION_KEYS"]
