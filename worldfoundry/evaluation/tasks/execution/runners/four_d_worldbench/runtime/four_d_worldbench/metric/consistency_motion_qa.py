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
                    "keye_answer":"",
                    "extract_answer":"",
                    "is_yes": False
                }


                messages = [
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text. Example: {\"answer\":\"yes\"} or {\"answer\":\"no\"}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,  # Pass video path
                            },
                            {
                                "type": "text", 
                                    "text": f"Strictly and only based on the video content, determine whether the described motion is rational.\n"
                                            f"- Visual-only: Use only what is visible in the frames; do not use prior/world knowledge, audio, subtitles, or assumptions.\n"
                                            f"- Temporal grounding: Base the judgment on the entire visible time span; consider order, continuity, direction, and duration of motion.\n"
                                            f"- Evidence sufficiency: If any required element is not visible, occluded, out of frame, too small, blurred, too fast, or temporally unclear, answer no.\n"
                                            f"- Consistency: If any frame contradicts the description (e.g., inconsistent direction/speed/trajectory/contact/collision), answer no.\n"
                                            f"- Multi-part descriptions: All listed conditions must be satisfied by the video; if any part fails or is missing, answer no.\n"
                                            f"- Ambiguity/noise: If the evidence is ambiguous, low-quality, or insufficient to be certain, answer no.\n"
                                            f"- Relevance: If the video content is unrelated to the described motion or shows a different action, answer no.\n"
                                            f"- Output format: Reply with exactly one lowercase word: yes or no. No punctuation, no explanations, nothing else.\n"
                                            f"- Default rule: When uncertain for any reason, answer no.\n"
                                            f"Question: {base_question[i]}\n"
                                            f"/think"
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
                    max_new_tokens=max_new_tokens,
                )

# Extract generated tokens (excluding input tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                raw_answer = output_text[0]
                answer = extract_yes_no_answer(raw_answer)
                print(f"Keye answer: {raw_answer}")
                question_detail["keye_answer"] = raw_answer
                question_detail["extract_answer"] = answer
                question_detail["is_yes"] = answer == "yes"
                if question_detail["is_yes"]:
                    score += 1
                    print(f"Answer is yes")
                
                video_detail["question_details"].append(question_detail)

            video_score = (score / question_num) if question_num > 0 else 0
            final_score += video_score
            
            video_detail["video_results"] = video_score
            video_num += 1
            processed_json.append(video_detail)
            
    ###  ?
    #average_score = final_score / video_num 

    return processed_json

def compute_consistency_motion_qa_metrics(json_dir, device, submodules_dict, **kwargs):  
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='consistency_motion_qa', lang='en')

    model = AutoModel.from_pretrained(
    keye_model_path(),
        torch_dtype="auto",
        device_map=device,
        attn_implementation=resolve_transformers_attention_implementation(device=device),
        trust_remote_code=True,
    )
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(keye_model_path(), min_pixels=min_pixels,max_pixels=max_pixels,trust_remote_code=True)

    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    video_details = Keye2_5VL_Video(prompt_dict_ls, model,processor,fps=2,device=device)
    final_average_score = (
        sum(d["video_results"] for d in video_details) / len(video_details)
        if video_details
        else 0.0
    )
    output_dir = os.path.dirname(json_dir)
    dim_name = os.path.splitext(os.path.basename(json_dir))[0]
    model_name = kwargs.get('model', '')
    dataset_json = kwargs.get('dataset_json', '')
    dataset_base = os.path.splitext(os.path.basename(dataset_json))[0] if dataset_json else 'dataset'
    suffix = f"{dim_name}__{model_name}__{dataset_base}_results.json" if model_name else f"{dim_name}_results.json"
    output_file = os.path.join(output_dir, suffix) 

    detailed_output = {
        "evaluation_summary": {
            "total_videos": len(video_details),
            "passed_videos": sum([d['video_results'] for d in video_details]),
            "average_score": final_average_score,
        },
        "video_details": video_details
    }   

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file}")
    return final_average_score, video_details 
