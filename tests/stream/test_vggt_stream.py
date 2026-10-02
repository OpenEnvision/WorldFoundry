import sys
from pathlib import Path

sys.path.append("..")

from worldfoundry.pipelines.vggt.pipeline_vggt import VGGTPipeline


DATA_PATH = "./data/test_cases/test_image_case1/ref_image.png"
MODEL_PATH = "facebook/VGGT-1B"
OUTPUT_DIR = "./vggt_stream_output"

POINT_CONF_THRESHOLD = 0.2
RESOLUTION = 518
PREPROCESS_MODE = "crop"
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 480
FPS = 12

# Unified 3D interaction schema, aligned with VGGTOperator
AVAILABLE_INTERACTIONS = [
    "forward", "backward", "left", "right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "camera_up", "camera_down",
    "camera_l", "camera_r",
    "camera_ul", "camera_ur", "camera_dl", "camera_dr",
    "camera_zoom_in", "camera_zoom_out",
]


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = VGGTPipeline.from_pretrained(
        representation_path=MODEL_PATH,
    )

    print("\n" + "=" * 40)
    print("VGGT Two-Stage 3DGS - Interactive Stream")
    print("=" * 40)
    print("Available interactions:", ", ".join(AVAILABLE_INTERACTIONS))
    print("Tips:")
    print("  - Input format: 'action1, action2' (e.g., 'forward, camera_l')")
    print("  - Input 'q' / 'quit' / 'n' to stop.")
    print("-" * 40)

    turn_idx = 0

    while True:
        user_input = input(f"\n[Turn {turn_idx}] Enter interactions: ").strip().lower()
        if user_input in ("q", "quit", "n"):
            print("Stopping interaction loop...")
            break
        if not user_input:
            continue

        raw_actions = [s.strip() for s in user_input.split(",") if s.strip()]
        valid_actions = []
        invalid_actions = []
        for act in raw_actions:
            if act in AVAILABLE_INTERACTIONS:
                valid_actions.append(act)
            else:
                invalid_actions.append(act)

        if invalid_actions:
            print(f"Warning: ignoring invalid interactions: {invalid_actions}")
            print(f"Please choose from: {AVAILABLE_INTERACTIONS}")
        if not valid_actions:
            print("No valid interactions provided. Try again.")
            continue

        print(f"Processing Turn {turn_idx}: interactions={valid_actions}")

        output_name = f"vggt_stream_turn_{turn_idx:03d}.mp4"
        output_video_path = pipeline(
            image_path=DATA_PATH,
            interactions=valid_actions,
            task_type="vggt_two_stage_3dgs",
            output_dir=str(output_dir),
            point_conf_threshold=POINT_CONF_THRESHOLD,
            resolution=RESOLUTION,
            preprocess_mode=PREPROCESS_MODE,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            output_name=output_name,
            fps=FPS,
        )

        print(f"[Turn {turn_idx}] Saved video to: {output_video_path}")
        turn_idx += 1

    print("\nInteractive VGGT stream finished.")


if __name__ == "__main__":
    main()
