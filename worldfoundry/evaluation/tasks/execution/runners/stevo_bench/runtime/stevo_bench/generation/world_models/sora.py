import io
import os
import time
from pathlib import Path
from typing import Optional

from .base import WorldModelRunner


class SoraRunner(WorldModelRunner):
    """World-model adapter for OpenAI Sora via the openai SDK.

    Submits an async video-generation job, polls until completion, then
    downloads the resulting MP4 to output_path.

    The init_frame is passed as `input_reference`. Sora requires the reference
    image to exactly match the configured `size`, so the frame is resized via
    Pillow before upload (Pillow must be installed).

    Config keys (in addition to base keys):
        api_key_env    (str)  Name of the environment variable holding the API key.
                              Default: "SORA_API_KEY".
        model_id       (str)  Sora model identifier. Default: "sora-2".
        size           (str)  Output resolution as "WxH". Default: "1280x720".
        seconds        (int)  Video duration in seconds. Default: 8.
        poll_interval  (int)  Seconds between status polls. Default: 20.
        timeout        (int)  Max seconds to wait for one generation. Default: 900.
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        api_key_env = config.get("api_key_env", "SORA_API_KEY")
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise EnvironmentError(
                f"[{name}] Missing required environment variable: {api_key_env}"
            )
        self.model_id: str = config.get("model_id", "sora-2")
        self.size: str = config.get("size", "1280x720")
        self.seconds: int = int(config.get("seconds", 8))
        self.poll_interval: int = int(config.get("poll_interval", 20))
        self.timeout: int = int(config.get("timeout", 900))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        task_id: str,
        prompt: str,
        init_frame: Optional[Path],
        output_path: Path,
        camera_control: Optional[str] = None,
    ) -> bool:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            kwargs: dict = dict(
                model=self.model_id,
                prompt=prompt if prompt and prompt.strip() else "Camera remains stationary. No cuts, no scene change, do not add new entities.",
                size=self.size,
                seconds=self.seconds,
            )

            # Attach the conditioning frame, resized to match `size`
            ref_file = None
            if init_frame and init_frame.exists():
                ref_file = self._prepare_reference(init_frame)
                kwargs["input_reference"] = ref_file

            # Submit (returns immediately with status='queued')
            video = client.videos.create(**kwargs)
            print(f"[{self.name}] Submitted {task_id}: id={video.id} status={video.status}")

            # Poll until terminal state
            elapsed = 0
            while video.status not in ("completed", "failed", "cancelled"):
                time.sleep(self.poll_interval)
                elapsed += self.poll_interval
                video = client.videos.retrieve(video.id)
                print(
                    f"[{self.name}] {task_id}: status={video.status} "
                    f"elapsed={elapsed}s"
                )
                if elapsed >= self.timeout:
                    print(
                        f"[{self.name}] Timeout ({elapsed}s) waiting for task {task_id}"
                    )
                    return False

            if ref_file is not None:
                ref_file.close()

            if video.status != "completed":
                failure_msg = getattr(video, "failure_message", None) or getattr(video, "error", None)
                print(
                    f"[{self.name}] Generation failed for {task_id}: "
                    f"status={video.status}"
                    + (f" | {failure_msg}" if failure_msg else "")
                )
                return False

            # Download and save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            content = client.videos.download_content(video.id, variant="video")
            content.write_to_file(str(output_path))

            print(f"[{self.name}] Saved {task_id} → {output_path}")
            return True

        except Exception as e:
            print(f"[{self.name}] Error generating task {task_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_reference(self, init_frame: Path) -> io.IOBase:
        """Return a file-like object for init_frame resized to self.size.

        Sora requires the input_reference image to exactly match the output
        resolution. Pillow is used to resize; the result is returned as an
        in-memory PNG buffer so no temporary files are written to disk.

        Raises ImportError if Pillow is not installed.
        """
        from PIL import Image

        w_str, h_str = self.size.split("x")
        target_w, target_h = int(w_str), int(h_str)

        img = Image.open(init_frame).convert("RGB")
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)

        buf = io.BytesIO()
        buf.name = init_frame.stem + ".png"   # name drives MIME detection in SDK
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
