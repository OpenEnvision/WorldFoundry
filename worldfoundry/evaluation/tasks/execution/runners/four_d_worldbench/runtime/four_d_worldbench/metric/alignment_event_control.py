from keye_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer

import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
import argparse
import os
from PIL import Image
from .utils import load_dimension_info
from ..paths import keye_model_path
from worldfoundry.core.attention import resolve_transformers_attention_implementation
from worldfoundry.core.utils import extract_yes_no_answer, resolve_generation_max_new_tokens
from tqdm import tqdm

# Import compatibility tools
try:
    from metric_compatibility import ensure_auxiliary_info_compatibility
    COMPATIBILITY_AVAILABLE = True
except ImportError:
    COMPATIBILITY_AVAILABLE = False
    
warnings.filterwarnings("ignore")

def Keye2_5VL_Video(prompt_dict_ls,model,processor,fps,device):
    final_score = 0
    valid_num = 0
    video_num = 0
    processed_json = []
    max_new_tokens = resolve_generation_max_new_tokens(512, scope="4d_worldbench")

    for prompt_dict in tqdm(prompt_dict_ls):
        base_question = prompt_dict['auxiliary_info']
        question_num = len(base_question)
        video_paths = prompt_dict['video_list']

        
        for video_path in video_paths:
            print(f"\n===== Start processing video: {video_path} =====")

            # Initialize detailed results for current video
            video_detail = {
                "video_path": video_path,
                "video_results": 0,
                "question_details": []
            }
            score = 0
            vision_inputs = None
            for i in range(len(base_question)):
                print(f"\n--- Processing question {i+1}/{len(base_question)} ---")
                print(f"Question: {base_question[i]}")

                # Initialize detailed information for current question
                question_detail = {
                    "question": base_question[i],
                    "extract_answer":"",
                    "is_yes": False
                }

                prompt_text = f"""Watch the video carefully and answer the following question about the specific plots showed in the video:

Question: {base_question[i]}

Instructions:
1. Carefully observe the video content and compare it with the question
2. If there is anything in the question that doesn't match what you observe in the video, answer "no"
3. Only answer "yes" if the question accurately describes what happens in the video
4. Think step by step if needed, but always end your response with a clear answer

Please provide your analysis and conclude with:
Answer: [Yes/No]"""  

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,  # Pass video path
                            },
                            {
                                "type": "text", 
                                "text": prompt_text
                            }
                        ]
                    }
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                if vision_inputs is None:
                    vision_inputs = process_vision_info(messages)
                image_inputs, video_inputs, mm_processor_kwargs = vision_inputs

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **mm_processor_kwargs,
                    fps=fps,
                )

                inputs = inputs.to(device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens
                )

# Extract generated tokens (excluding input tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(f"Model answer: {output_text}")
                question_detail["keye_answer"] = output_text
                question_detail["extract_answer"] = extract_yes_no_answer(output_text)
                print(f"Extracted answer: {question_detail['extract_answer']}")
                question_detail["is_yes"] = question_detail["extract_answer"] == "yes"
                #question_detail["is_yes"] = "yes" in output_text.lower()

                if question_detail["is_yes"]:
                    score += 1
                video_detail["question_details"].append(question_detail)

            video_score = score / question_num if question_num else 0.0
            final_score += video_score
            
            video_detail["video_results"] = video_score
            video_num += 1
        
            processed_json.append(video_detail)
        
    average_score = final_score / video_num if video_num > 0 else 0

    return average_score,processed_json

def compute_complex_plot(json_dir, device, submodules_dict, **kwargs):  
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='alignment_event_control', lang='en')
    model_name = kwargs.get('model', '')
    dataset_json = kwargs.get('dataset_json', '')
    keye_model = AutoModel.from_pretrained(
    keye_model_path(),
        torch_dtype="auto",
        device_map=device,
        attn_implementation=resolve_transformers_attention_implementation(device=device),
        trust_remote_code=True,
    )
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(keye_model_path(), min_pixels=min_pixels,max_pixels=max_pixels,trust_remote_code=True)

    keye_model = keye_model.to(device)
    keye_model.eval()

    print("Model loaded successfully")
    average_results,video_details = Keye2_5VL_Video(prompt_dict_ls, keye_model,processor,fps=1,device=device)
    final_average_score = (
        sum(d["video_results"] for d in video_details) / len(video_details)
        if video_details
        else 0.0
    )

    output_dir = os.path.dirname(json_dir)
    dim_name = os.path.splitext(os.path.basename(json_dir))[0]
    dataset_base = os.path.splitext(os.path.basename(dataset_json))[0] if dataset_json else 'dataset'
    suffix = f"{dim_name}__{model_name}__{dataset_base}_results.json" if model_name else f"{dim_name}_results.json"
    output_file = os.path.join(output_dir, suffix)

    detailed_output = {
        "evaluation_summary": {
            "total_videos": len(video_details),
            "total_score": sum([d['video_results'] for d in video_details]),
            "average_score": final_average_score
        },
        "video_details": video_details
    }   

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file}")
    return average_results, video_details    
