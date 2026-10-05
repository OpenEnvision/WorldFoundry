import os
import time
from pathlib import Path
from typing import Optional

from .base import WorldModelRunner


class VeoRunner(WorldModelRunner):
    """World-model adapter for Google Veo via the google-genai SDK.

    Submits an async video-generation job, polls until completion, then
    downloads the resulting MP4 to output_path.

    Config keys (in addition to base keys):
        api_key_env    (str)  Name of the environment variable holding the API key.
                              Default: "GOOGLE_API_KEY".
        model_id       (str)  Veo model identifier.
                              Default: "veo-3-0-generate-001".
        poll_interval  (int)  Seconds to wait between polling attempts. Default: 10.
        timeout        (int)  Max seconds to wait for a single generation. Default: 600.
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        api_key_env = config.get("api_key_env", "GOOGLE_API_KEY")
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise EnvironmentError(
                f"[{name}] Missing required environment variable: {api_key_env}"
            )
        self.model_id: str = config.get("model_id", "veo-3.1-fast-generate-preview")
        self.poll_interval: int = int(config.get("poll_interval", 10))
        self.timeout: int = int(config.get("timeout", 600))
        self.seconds: int = int(config.get("seconds", 8))

    def generate(
        self,
        task_id: str,
        prompt: str,
        init_frame: Optional[Path],
        output_path: Path,
        camera_control: Optional[str] = None,
    ) -> bool:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            # Build optional conditioning image
            image = None
            if init_frame and init_frame.exists():
                mime_type = (
                    "image/png" if init_frame.suffix.lower() == ".png" else "image/jpeg"
                )
                image = types.Image(
                    image_bytes=init_frame.read_bytes(),
                    mime_type=mime_type,
                )

            # Submit async generation job
            operation = client.models.generate_videos(
                model=self.model_id,
                prompt=prompt,
                image=image,
                config=types.GenerateVideosConfig(
                    number_of_videos=1,
                    aspect_ratio="16:9",
                    resolution="720p",
                    duration_seconds=self.seconds,
                ),
            )

            # Poll until done or timeout
            elapsed = 0
            while not operation.done:
                time.sleep(self.poll_interval)
                elapsed += self.poll_interval
                operation = client.operations.get(operation)
                if elapsed >= self.timeout:
                    print(
                        f"[{self.name}] Timeout ({elapsed}s) waiting for task {task_id}"
                    )
                    return False

            # Check for an API-level error (content policy rejection, bad params, etc.)
            if operation.error:
                print(
                    f"[{self.name}] Generation failed for {task_id}: {operation.error}"
                )
                return False

            videos = (
                operation.response.generated_videos
                if operation.response
                else None
            )
            if not videos:
                print(
                    f"[{self.name}] No videos returned for {task_id} "
                    f"(response: {operation.response})"
                )
                return False

            generated_video = videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(str(output_path))

            return True

        except Exception as e:
            print(f"[{self.name}] Error generating task {task_id}: {e}")
            return False
