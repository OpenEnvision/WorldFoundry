import copy
import warnings

from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.conversation import conv_templates
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.mm_utils import (
    tokenizer_image_token,
)
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.model.builder import load_pretrained_model

# isort: split
from vbench2.utils import load_dimension_info

# isort: split
from tqdm import tqdm

from worldfoundry.core.execution.inference_runtime import resolve_generation_max_new_tokens
from worldfoundry.core.media.codecs.video import sample_video_frames
from worldfoundry.core.platform.device import resolve_inference_dtype
from worldfoundry.core.utils import extract_yes_no_answer

warnings.filterwarnings("ignore")

load_video = sample_video_frames


def LLaVA_Video(prompt_dict_ls, model, tokenizer, image_processor, device):
    final_score = 0
    valid_num = 0
    processed_json = []
    inference_dtype = resolve_inference_dtype(device)
    max_new_tokens = resolve_generation_max_new_tokens(512, scope="vbench2")
    for prompt_dict in tqdm(prompt_dict_ls):
        base_question = prompt_dict["auxiliary_info"]
        video_paths = prompt_dict["video_list"]
        for video_path in video_paths:
            max_frames_num = 64
            video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(
                device=device, dtype=inference_dtype, non_blocking=True
            )
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            flag = True
            for i in range(len(base_question)):
                if i == 0:
                    videoa = [video[:32]]
                    first_half_times = ",".join(frame_time.split(",")[: len(videoa[0])])
                    time_instruciton = f"The video lasts for {video_time / 2:.2f} seconds, and {len(videoa[0])} frames are uniformly sampled from it. These frames are located at {first_half_times}. Please answer yes or no only for the following questions."
                    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{base_question[i]}"
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    input_ids = (
                        tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .to(device)
                    )
                    cont = model.generate(
                        input_ids,
                        images=videoa,
                        modalities=["video"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=max_new_tokens,
                    )
                elif i == 1:
                    videoa = [video[-1].unsqueeze(0)]
                    time_instruciton = "Please answer yes or no only for the following questions."
                    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{base_question[i]}"
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    input_ids = (
                        tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .to(device)
                    )
                    cont = model.generate(
                        input_ids,
                        images=videoa,
                        modalities=["image"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    videoa = [video]
                    time_instruciton = "Please answer yes or no only for the following question."
                    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{base_question[i]}"
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    input_ids = (
                        tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .to(device)
                    )
                    cont = model.generate(
                        input_ids,
                        images=videoa,
                        modalities=["video"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=max_new_tokens,
                    )
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                if extract_yes_no_answer(text_outputs) != "yes":
                    flag = False
            if flag:
                final_score += 1
                sco = 1
            else:
                sco = 0
            valid_num += 1
            processed_json.append({"video_path": video_path, "video_results": sco})

    return (final_score / valid_num if valid_num else 0.0), processed_json


def compute_dynamic_spatial_relationship(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension="dynamic_spatial_relationship", lang="en")

    model_name = "llava_qwen"
    device_map = "auto"
    try:
        pretrained = submodules_dict["llava"]
        llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name, torch_dtype=resolve_inference_dtype(device), device_map=device_map
        )  # Add any other thing you want to pass in llava_model_args
    except BaseException:
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name, torch_dtype=resolve_inference_dtype(device), device_map=device_map
        )  # Add any other thing you want to pass in llava_model_args
    llava_model.eval()

    all_results, video_results = LLaVA_Video(prompt_dict_ls, llava_model, llava_tokenizer, image_processor, device)
    all_results = sum(d["video_results"] for d in video_results) / len(video_results) if video_results else 0.0
    return all_results, video_results
