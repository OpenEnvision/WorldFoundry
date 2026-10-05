import os
import json
from glob import glob
import cv2
import torch
from PIL import Image
import argparse
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.reproducibility import (
    initialize_reproducibility,
)

class CaptionModel:
    def __init__(self, model_path="microsoft/Florence-2-large", batch_size=8):
        print(f"Loading Florence-2 model from {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            attn_implementation="eager"  # prevents _supports_sdpa errors with some model versions
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
        )

        self.batch_size = batch_size
        self.task_prompt = "<MORE_DETAILED_CAPTION>"

    def process_batch(self, image_list):
        if not image_list:
            return []

        prompts = [self.task_prompt] * len(image_list)
        inputs = self.processor(text=prompts, images=image_list, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            use_cache=False  # disable KV cache for compatibility with older model code
        )

        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)

        final_captions = []
        for text, img in zip(generated_texts, image_list):
            text = text.replace("<pad>", "")
            try:
                parsed_answer = self.processor.post_process_generation(
                    text,
                    task=self.task_prompt,
                    image_size=(img.width, img.height)
                )
                caption_core = parsed_answer[self.task_prompt]
                final_captions.append(caption_core)
            except Exception as e:
                print(f"Post-process warning: {e}")
                final_captions.append(text)

        return final_captions

def extract_middle_frame(vid_path):
    """Read the middle frame of a video."""
    try:
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None

        middle_idx = max(0, frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        success, frame = cap.read()
        cap.release()

        if success:
            return Image.fromarray(frame[..., ::-1])
        return None
    except Exception as e:
        print(f"Error processing video {vid_path}: {e}")
        return None

def process_videos(video_paths, vggt_root, model_path, batch_size):
    """Process a list of video paths and return caption results."""
    caption_model = CaptionModel(model_path=model_path, batch_size=batch_size)

    results = []
    batch_images = []
    batch_meta = []

    for vid_path in tqdm(video_paths, desc="Processing Videos"):
        pil_img = extract_middle_frame(vid_path)

        if pil_img is None:
            print(f"Skipping broken video: {vid_path}")
            continue

        vid_name = os.path.basename(vid_path).replace(".mp4", "")
        vggt_depth_path = os.path.join(vggt_root, vid_name)
        vggt_extrinsics_path = os.path.join(vggt_depth_path, "extrinsics.txt")

        entry_meta = {
            "video_path": vid_path,
            "vggt_depth_path": vggt_depth_path,
            "vggt_extrinsics_path": vggt_extrinsics_path,
            "radius_ratio": 1
        }

        batch_images.append(pil_img)
        batch_meta.append(entry_meta)

        if len(batch_images) >= batch_size:
            captions = caption_model.process_batch(batch_images)
            for meta, cap in zip(batch_meta, captions):
                meta["text"] = cap
                results.append(meta)
            batch_images = []
            batch_meta = []

    if batch_images:
        captions = caption_model.process_batch(batch_images)
        for meta, cap in zip(batch_meta, captions):
            meta["text"] = cap
            results.append(meta)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Input video folder containing .mp4 files")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_path", type=str, default="microsoft/Florence-2-large", help="Florence-2 model path (local or HuggingFace)")
    parser.add_argument("--worker_id", type=int, default=None, help="Worker index for multi-GPU parallel (0-based)")
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of workers for multi-GPU parallel")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path (default: <root_dir>/new.json)")
    args = parser.parse_args()
    initialize_reproducibility()

    root_dir = args.root_dir
    vggt_root = os.path.join(root_dir, "new_vggt")

    video_paths = sorted(glob(os.path.join(root_dir, "*.mp4")))
    print(f"Found {len(video_paths)} videos total.")

    # If running as a worker, only process assigned subset
    if args.worker_id is not None:
        video_paths = [v for i, v in enumerate(video_paths) if i % args.num_workers == args.worker_id]
        print(f"Worker {args.worker_id}/{args.num_workers}: processing {len(video_paths)} videos")

    results = process_videos(video_paths, vggt_root, args.model_path, args.batch_size)

    output_json = args.output_json or os.path.join(root_dir, "new.json")
    print(f"Saving {len(results)} captions to {output_json}...")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
