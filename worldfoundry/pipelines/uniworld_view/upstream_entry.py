"""Small compatibility entry for the pinned UniWorld-View upstream revision.

At 660212c4, UniScene.setup_stream3r is a no-op although dynamic_view calls
self.stream3r in _estimate_geometry_stream3r.  Restore the one upstream load
line. The single-view final video also omits fps; pass through the requested
CLI rate there so both tasks honor the public interface. The upstream
diffusion call hard-codes seed 42; honor the already exposed CLI --seed.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path.cwd()
    sys.path.insert(0, str(repo_root))
    # Match inference.py's import order: vendored dependencies imported by
    # demo also expose a top-level `configs` package.
    from configs.infer_config import get_parser as _get_parser

    import demo

    STream3R = demo.STream3R
    UniScene = demo.UniScene

    def setup_stream3r(self) -> None:
        if self.opts.mode == "dynamic_view":
            self.stream3r = STream3R.from_pretrained(self.opts.stream3r_path).to(self.opts.device).eval()
        else:
            self.stream3r = None

    UniScene.setup_stream3r = setup_stream3r
    seed_index = sys.argv.index("--seed") if "--seed" in sys.argv else None
    requested_seed = int(sys.argv[seed_index + 1]) if seed_index is not None else 43
    original_pipeline_call = demo.WanVACEPipeline.__call__

    def seeded_pipeline_call(self, *args, **kwargs):
        generator = kwargs.get("generator")
        generators = generator if isinstance(generator, list) else [generator]
        for item in generators:
            if isinstance(item, demo.torch.Generator):
                item.manual_seed(requested_seed)
        return original_pipeline_call(self, *args, **kwargs)

    demo.WanVACEPipeline.__call__ = seeded_pipeline_call
    # In the pinned single-view path, the final save_video call omits fps and
    # inherits the utility's 8 fps default, ignoring the CLI's --fps value.
    # Apply the requested rate only to the final artifact; dynamic_view already
    # passes opts.fps explicitly.
    fps_index = sys.argv.index("--fps") if "--fps" in sys.argv else None
    requested_fps = int(sys.argv[fps_index + 1]) if fps_index is not None else 16
    original_save_video = demo.save_video

    def save_video(data, images_path, folder=None, fps=None):
        if fps is None and Path(images_path).name == "diffusion_correct.mp4":
            fps = requested_fps
        if fps is None:
            return original_save_video(data, images_path, folder)
        return original_save_video(data, images_path, folder, fps=fps)

    demo.save_video = save_video
    runpy.run_path(str(repo_root / "inference.py"), run_name="__main__")


if __name__ == "__main__":
    main()
