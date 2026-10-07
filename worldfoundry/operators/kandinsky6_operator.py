"""Prompt and first-frame requests for Kandinsky-6 audio-video generation."""

from pathlib import Path

from .base_operator import BaseOperator


class Kandinsky6Operator(BaseOperator):
    def __init__(self):
        super().__init__(["textual_instruction", "visual_instruction"])

    def prepare(self, prompt, images=None, video=None, **options):
        from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Kandinsky-6 requires a non-empty prompt")
        if video is not None:
            raise ValueError("Kandinsky-6 accepts a first-frame image, not input video")
        image = options.pop("image", images)
        if isinstance(image, (list, tuple)):
            if len(image) != 1:
                raise ValueError("Kandinsky-6 requires exactly one first-frame image")
            image = image[0]
        if isinstance(image, (str, Path)):
            image = Path(image).expanduser().resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            image = str(image)
        width = int(options.pop("width", 864))
        height = int(options.pop("height", 480))
        frames = int(options.pop("num_frames", 121))
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("Kandinsky-6 width and height must be positive multiples of 16")
        if frames < 5 or (frames - 1) % 4:
            raise ValueError("Kandinsky-6 num_frames must be 4*n+1, with n >= 1")
        request = DiffusionRequest(
            prompt=prompt,
            negative_prompt=options.pop("negative_prompt", None),
            width=width,
            height=height,
            num_frames=frames,
            sampling=SamplingConfig(
                num_inference_steps=int(options.pop("num_inference_steps", 10)),
                guidance_scale=float(options.pop("guidance_scale", 1.0)),
                seed=int(options.pop("seed", 0)),
            ),
            inputs={
                "image": image,
                "sample_audio": bool(options.pop("sample_audio", True)),
                "save_path": options.pop("save_path", None),
            },
        )
        if options:
            raise TypeError("Unknown Kandinsky-6 generation options: " + ", ".join(sorted(options)))
        return request
