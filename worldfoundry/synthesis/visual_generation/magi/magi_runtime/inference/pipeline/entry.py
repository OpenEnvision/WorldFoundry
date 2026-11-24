# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
from typing import Any


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MagiPipeline with different modes.")
    parser.add_argument("--config_file", type=str, help="Path to the configuration file.")
    parser.add_argument(
        "--mode", type=str, choices=["t2v", "i2v", "v2v"], required=True, help="Mode to run: t2v, i2v, or v2v."
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the pipeline.")
    parser.add_argument("--image_path", type=str, help="Path to the image file (for i2v mode).")
    parser.add_argument("--prefix_video_path", type=str, help="Path to the prefix video file (for v2v mode).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video.")
    return parser.parse_args()


def _wrap_triton(kernel: Any) -> Any:
    return kernel


def main():
    args = parse_arguments()
    import torch

    if not hasattr(torch.library, "wrap_triton"):
        torch.library.wrap_triton = _wrap_triton

    from .pipeline import MagiPipeline

    pipeline = MagiPipeline(args.config_file)

    if args.mode == "t2v":
        pipeline.run_text_to_video(prompt=args.prompt, output_path=args.output_path)
    elif args.mode == "i2v":
        if not args.image_path:
            print("Error: --image_path is required for i2v mode.")
            sys.exit(1)
        pipeline.run_image_to_video(prompt=args.prompt, image_path=args.image_path, output_path=args.output_path)
    elif args.mode == "v2v":
        if not args.prefix_video_path:
            print("Error: --prefix_video_path is required for v2v mode.")
            sys.exit(1)
        pipeline.run_video_to_video(
            prompt=args.prompt, prefix_video_path=args.prefix_video_path, output_path=args.output_path
        )


if __name__ == "__main__":
    main()
