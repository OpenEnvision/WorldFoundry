# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the public DreamDojo loader with explicit checkpoints and normalized actions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--stream", action="store_true", help="Run the resident student configure/STEP/reset route")
    parser.add_argument("--vae", required=True, type=Path)
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text-encoder", type=Path)
    text.add_argument("--text-embeddings", type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.variant == "student" and args.student_checkpoint is None:
        parser.error("--variant student requires --student-checkpoint")
    if args.variant == "teacher" and (args.checkpoint_dir is None or args.stream):
        parser.error("teacher requires --checkpoint-dir and does not support --stream")

    from worldfoundry.pipelines.component_pipelines import DreamDojoPipeline
    from worldfoundry.pipelines.dreamdojo.pipeline_dreamdojo_student import DreamDojoStudentPipeline

    options = {"vae_path": args.vae}
    if args.text_encoder is not None:
        options["text_encoder_path"] = args.text_encoder
    else:
        options["text_embeddings_path"] = args.text_embeddings
    options.update(
        {"student_checkpoint_path": args.student_checkpoint}
        if args.variant == "student"
        else {"checkpoints_dir": args.checkpoint_dir}
    )
    pipeline_class = DreamDojoStudentPipeline if args.variant == "student" else DreamDojoPipeline
    pipeline = pipeline_class.from_pretrained(options, device="cuda")
    steps = args.steps if args.steps is not None else (4 if args.variant == "student" else 35)
    if args.stream:
        import torch

        from worldfoundry.base_models.diffusion_model.models.encoders.dreamdojo import robot_actions
        from worldfoundry.core.io import write_json
        from worldfoundry.core.media.codecs.video import save_image_or_video_tensor

        actions = robot_actions(args.actions)
        if actions.shape[1] < 12 or actions.shape[1] % 4:
            parser.error("student actions require 12 initial actions followed by complete 4-action chunks")
        student = cast(DreamDojoStudentPipeline, pipeline)
        student.configure_realtime(args.image, seed=args.seed, num_inference_steps=steps)
        chunks = []
        try:
            for start, end in [(0, 12), *((index, index + 4) for index in range(12, actions.shape[1], 4))]:
                chunk = student.stream_realtime(actions=actions[:, start:end])
                chunks.append(chunk["video_chunk"].frames.detach().cpu())
        finally:
            student.reset_realtime()
        pixels = torch.cat(chunks, dim=2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_image_or_video_tensor(pixels, str(args.output), fps=10)
        report = {
            "model_id": "dreamdojo-student",
            "student_checkpoint_path": str(args.student_checkpoint),
            "artifact_path": str(args.output),
            "num_frames": pixels.shape[2],
            "fps": 10,
            "num_inference_steps": steps,
            "seed": args.seed,
            "route": "resident-student-stream",
        }
        write_json(args.output.with_suffix(".json"), report)
        print(report)
        return
    result = pipeline(
        images=args.image,
        interactions=[args.actions],
        output_path=args.output,
        num_inference_steps=steps,
        seed=args.seed,
        return_dict=True,
    )
    print({key: value for key, value in result.items() if key != "video_chunk"})


if __name__ == "__main__":
    main()
