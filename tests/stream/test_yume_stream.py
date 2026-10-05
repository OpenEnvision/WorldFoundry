from diffusers.utils import export_to_video
from PIL import Image
from worldfoundry.pipelines.yume.pipeline_yume import YumePipeline
import torch


prompt = "A fire-breathing dragon appeared."
image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

pretrained_model_path = "stdstu123/Yume-I2V-540P"
pipeline = YumePipeline.from_pretrained(
    model_path=pretrained_model_path,
    device="cuda",
    weight_dtype=torch.bfloat16,
    fsdp=True
)

AVAILABLE_INTERACTIONS = [
    "forward", "left", "right", "backward",
    "camera_l", "camera_r", "camera_up", "camera_down"
]

print("Available interactions:")
for i, interaction in enumerate(AVAILABLE_INTERACTIONS):
    print(f"  {i + 1}. {interaction}")
print("Tips:")
print("  - You can input multiple interactions separated by comma (e.g., 'forward,camera_l')")
print("  - Input 'n' or 'q' to stop and export video")

print("--- Interactive Stream Started ---")
turn_idx = 0

while True:
    interaction_input = input(f"\n[Turn {turn_idx}] Enter interaction(s) (or 'n'/'q' to stop): ").strip().lower()

    if interaction_input in ["n", "q"]:
        print("Stopping interaction loop...")
        break

    current_signal = [s.strip() for s in interaction_input.split(",") if s.strip()]

    invalid_signals = [s for s in current_signal if s not in AVAILABLE_INTERACTIONS]
    if invalid_signals:
        print(f"Invalid interaction(s): {invalid_signals}")
        print(f"Please choose from: {AVAILABLE_INTERACTIONS}")
        continue

    if not current_signal:
        print("No valid interaction provided. Please try again.")
        continue

    try:
        speed = float(input(f"[Turn {turn_idx}] Enter interaction speed (e.g., '4'): ").strip())
        distance = float(input(f"[Turn {turn_idx}] Enter movement distance (e.g., '4'): ").strip())
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        continue

    interaction_speeds = [speed] * len(current_signal)
    interaction_distances = [None if s.startswith("camera_") else distance for s in current_signal]

    print(
        f"Processing turn {turn_idx} with signals: {current_signal}, "
        f"speeds: {interaction_speeds}, distances: {interaction_distances}"
    )

    start_img = input_image if turn_idx == 0 else None

    video_output = pipeline.stream(
        prompt=prompt,
        interactions=current_signal,
        interaction_speeds=interaction_speeds,
        interaction_distances=interaction_distances,
        images=start_img,
        size="544*960",
        seed=43,
        task_type="i2v"
    )

    turn_idx += 1
    print(f"Frames generated in this turn: {len(video_output)}, Total frames: {len(pipeline.memory_module.all_frames)}")

print(f"Total frames generated: {len(pipeline.memory_module.all_frames)}")

if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
    export_to_video(pipeline.memory_module.all_frames, "yume_stream_demo.mp4", fps=16)
    print("Video saved successfully.")
