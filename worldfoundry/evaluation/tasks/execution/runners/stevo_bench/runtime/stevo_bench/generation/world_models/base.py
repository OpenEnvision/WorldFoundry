from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class WorldModelRunner(ABC):
    """Abstract interface that every world-model adapter must implement.

    Each concrete subclass handles one model's invocation details (API calls,
    subprocess commands, authentication, retries, etc.) and exposes a single
    `generate()` method that the orchestrator calls uniformly.

    Config keys consumed by every runner (from models.yaml):
        prompt_field         (str)            Which prompts.* YAML key to use as
                                              the text prompt (e.g. "video_WM").
        camera_control_field (str | null)     Which camera_control.* YAML key to
                                              extract and pass along (or None).
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.prompt_field: str = config.get("prompt_field", "video_WM")
        self.camera_control_field: Optional[str] = config.get("camera_control_field")

    @abstractmethod
    def generate(
        self,
        task_id: str,
        prompt: str,
        init_frame: Optional[Path],
        output_path: Path,
        camera_control: Optional[str] = None,
    ) -> bool:
        """Generate a video for one task.

        Args:
            task_id:        Identifier used only for logging.
            prompt:         Text prompt, already extracted from the correct YAML field.
            init_frame:     Conditioning image path, or None if not available / needed.
            output_path:    Where to write the resulting .mp4 file.
            camera_control: Model-specific camera-control string (e.g. "s-3, right-30,
                            left-30"), or None if this model has no camera_control_field.

        Returns:
            True if the video was successfully written to output_path, False otherwise.
        """
        ...
