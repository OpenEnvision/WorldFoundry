import json
import os

import torch
from EWMBench.utils import load_dimension_info
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.qwen_vl_utils import process_vision_info
from worldfoundry.core.execution.device import get_current_torch_device, resolve_inference_dtype
from worldfoundry.core.execution.inference_runtime import resolve_generation_max_new_tokens
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    import json_repair
except ImportError:
    json_repair = json


def inference(
    model,
    processor,
    video_path,
    prompt,
    max_new_tokens=None,
    total_pixels=20480 * 28 * 28,
    min_pixels=16 * 28 * 28,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant in analyzing videos."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs["fps"]
    if isinstance(fps_inputs, (list, tuple)):
        fps_inputs = fps_inputs[0] if fps_inputs else None
    max_new_tokens = max_new_tokens or resolve_generation_max_new_tokens(512, scope="worldarena_caption")
    try:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=fps_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        output_text = ["Error: " + str(e)]

    return output_text[0]


def prepare_prompt(video_path):
    return """
   Analyze the video captured by the overhead camera mounted on a robotic and perform the following tasks:
   
   NOTED: There should be **no human hands** in the video. These are often wrongly generated from the robot's gripper. If human hands appear, note them as **abnormal** and a **violation of logical constraints**.

   1. Describe the video in general:
      - Provide a brief description on what tasks the robotic is performing. When something anomaly happens, pay special attention to it.

   2. Describe the events:
      - Provide a detailed, step-by-step description of all actions in the video, focusing on their sequence.

   3. Identify key events with logical constraints:
      - Extract critical actions subject to logical rules, such as:
      - Prerequisites (e.g., opening a door before accessing items).
      - Avoiding physical violations (e.g., objects passing through barriers).
      - Maintaining logical task order.
      - Identify any violations of these constraints.
      
   Provide the result in json format with each key representing a task.

   {
   "General": Brief description on tasks the robotic is performing.
   "Events": Chronological list of all actions.
   "Logical_Constraints": Key actions, their constraints, and whether they are satisfied.
   "Overall_Constraints": True if the logical constraints are satisfied, false otherwise.
   }

   """


def caption_reference(model_name, model_path, video_folder_root, save_path, **kwargs):
    device = get_current_torch_device()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=resolve_inference_dtype(device),
        device_map=str(device),
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    json_file_name = f"{model_name}_caption_responses.json"
    json_file_name = os.path.join(save_path, json_file_name)

    if not os.path.exists(json_file_name):
        all_mp4_files = load_dimension_info(video_folder_root, dimension="semantics")

        all_responses = {}
        for mp4_file in tqdm(all_mp4_files):
            prompt = prepare_prompt(mp4_file)
            try:
                response = json_repair.loads(inference(model, processor, mp4_file, prompt))
            except Exception as e:
                print(f"Error: {str(e)}")
                response = "Error: " + str(e)

            parts = mp4_file.split("/")
            model_dataset = f"{model_name}_dataset"
            try:
                start_index = parts.index(model_dataset) + 1
                video_index = parts.index("video")
                selected_parts = parts[start_index:video_index]
                mp4_file_name = "_".join([model_dataset] + selected_parts)
            except ValueError:
                mp4_file_name = "error_in_filename_construction"

            all_responses[mp4_file_name] = response

        with open(json_file_name, "w") as f:
            json.dump(all_responses, f, indent=4)
