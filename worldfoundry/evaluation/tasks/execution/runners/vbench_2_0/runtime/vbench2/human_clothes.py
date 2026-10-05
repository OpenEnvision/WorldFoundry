from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.model.builder import load_pretrained_model
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
import json
import os
import argparse
from vbench2.utils import load_dimension_info
from tqdm import tqdm
from worldfoundry.core.io import sample_video_frames
from worldfoundry.core.execution.device import resolve_inference_dtype
from worldfoundry.core.utils import extract_yes_no_answer, resolve_generation_max_new_tokens

warnings.filterwarnings("ignore")

load_video = sample_video_frames


def LLaVA_Video(prompt_dict_ls, model, tokenizer, image_processor, device):
    final_score = 0
    valid_num = 0
    processed_json=[]
    inference_dtype = resolve_inference_dtype(device)
    max_new_tokens = resolve_generation_max_new_tokens(512, scope="vbench2")
    base_question=["Is there only one person in the video throughout?", "Is the person in the video the same throughout?", "Does the clothes of the person in the video (color, texture) remain consistent throughout?"]
    for prompt_dict in tqdm(prompt_dict_ls):
        question_num = len(base_question)
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
        
            max_frames_num = 64
            video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(
                device=device, dtype=inference_dtype, non_blocking=True
            )
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            video=[video]
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Return yes or no only for the following question."
            score=0
            flag=True
            valid=True
            new_item = {
                "video_path": video_path,
            }
            for i in range(len(base_question)):
                question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{base_question[i]}"
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                cont = model.generate(
                        input_ids,
                        images=video,
                        modalities= ["video"],
                        do_sample=False,
                        temperature=0,
                    max_new_tokens=max_new_tokens,
                    )
            
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                
                if i==0 and extract_yes_no_answer(text_outputs) != "yes":
                    valid=False
                    break
                elif i!=0 and extract_yes_no_answer(text_outputs) != "yes":
                    flag=False
            if not valid:
                new_item[f"video_results"]=-1
                processed_json.append(new_item)
                continue 
            if flag:
                final_score+=1
                new_item[f"video_results"]=1
            else:
                new_item[f"video_results"]=0
            valid_num+=1

            processed_json.append(new_item)
    return (final_score / valid_num if valid_num else 0.0), processed_json
        
        
def compute_human_clothes(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='human_clothes', lang='en')
    model_name = "llava_qwen"
    device_map = "auto"
    try:
        pretrained = submodules_dict['llava']
        llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype=resolve_inference_dtype(device), device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    except:
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype=resolve_inference_dtype(device), device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    llava_model.eval()
    
    all_results, video_results = LLaVA_Video(prompt_dict_ls, llava_model, llava_tokenizer, image_processor, device)
    score=0
    num=0
    for d in video_results:
        if d['video_results']!=-1:
            num+=1
            score+= d['video_results']
    all_results = score / num if num else 0.0
    return all_results, video_results
