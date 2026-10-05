import ast
import os
import json
import fire
import random
import numpy as np
import prettytable as pt
import regex as re
from pathlib import Path
from datasets import load_dataset

aspects = [
    "visual quality",
    "temporal consistency",
    "dynamic degree",
    "text-to-video alignment",
    "factual consistency"
]

def get_pairwise_acc(left_score, right_score, left_preference, right_preference):
    if left_preference > right_preference:
        return left_score > right_score
    elif left_preference < right_preference:
        return left_score < right_score
    else:
        return abs(left_score - right_score) < 0.5

def count_total_pairs(data_map):
    return sum([sum([len(item["preference"][left_model_name]) for left_model_name in item["preference"]]) for item in data_map.values()])

def get_all_json_files(results_dir):
    results_dir = Path(results_dir)
    files = []
    for file in results_dir.iterdir():
            if file.is_file() and file.suffix == ".json":
                files.append(str(file))
            elif file.is_dir():
                files.extend(get_all_json_files(file))
    return files


def main(
    data_repo_name: str="TIGER-Lab/VideoScore-Bench",
    result_dir: str="./eval_results/vbench",
    csv_dir:str="./eval_results/vbench/csv_results",
    exclude_tie:bool=False,
    seed: int=42,
    
):
    random.seed(seed)
        
    ref_data=load_dataset(data_repo_name,name="vbench",split="test")
    
    source_list = ["technical_quality","subject_consistency","dynamics_degree","motion_smoothness","overall_consistency"]
    for target_source_idx, source_name in enumerate(source_list):
        print(f"-------------------{source_name}-------------------")
        target_source_idx = source_list.index(source_name)        
        
        ref_data_map = {}
        for idx,item in enumerate(ref_data):
            
            item["preference"]={k:v for k,v in item["preference"].items() if v is not None}

            # source format: technical_quality_0_cogvideo
            # target format: technical_quality-0
            idx = item["id"].split("_")[0] + "_" + item["id"].split("_")[1] + "-" + str(int(item["id"].split("_")[2]))
            model_name = item["id"].split("_")[3]
            aspect_name = item["id"].split("_")[0] + "_" + item["id"].split("_")[1]
            if aspect_name != source_name:
                continue
            # For this video, the text prompt is "(.*)"
            text_prompt = re.search("For this video, the text prompt is \"(.*)\"", item["conversations"][0]["value"]).group(1)
            if idx not in ref_data_map:
                ref_data_map[idx] = {"preference": {}, "prompt": text_prompt}
            if not exclude_tie:
                ref_data_map[idx]["preference"][model_name] = item['preference']
            else:
                ref_data_map[idx]["preference"][model_name] = {k: v for k, v in item['preference'].items() if v != 0.5}
        
        all_model_acc = {}
        
        # add random as a model
        random_acc = 0
        for idx, item in ref_data_map.items():
            model_scores = {model_name: random.random() * 4 for model_name in item["preference"]}
            for left_model_name in item["preference"]:
                for right_model_name in item["preference"][left_model_name]:
                    left_score = model_scores[left_model_name]
                    right_score = model_scores[right_model_name]
                    left_preference = item["preference"][left_model_name][right_model_name]
                    right_preference = 1 - left_preference
                    if get_pairwise_acc(left_score, right_score, left_preference, right_preference):
                        random_acc += 1
                        
        total_pairs = sum([sum([len(item["preference"][left_model_name]) for left_model_name in item["preference"]]) for item in ref_data_map.values()])
        random_acc = round(random_acc/total_pairs, 4)
        all_model_acc["Random"] = {
            "acc": random_acc,
            "aspect_accs": {aspect: random_acc for aspect in aspects},
            "total_examples": total_pairs,
        }
        
        for res_file in sorted(os.listdir(result_dir)):
            if not res_file.startswith("eval_") or not res_file.endswith(".json"):
                continue
            print(f"Processing {res_file}")
            file_model_name = res_file.split(".")[0].split("_")[-1]
            data = json.load(open(f"{result_dir}/{res_file}", 'r'))
            
            data_map = {}
            last_idx = -1
            source_idx = 0
            for item in data:
                # "dynamics_degree_27_cogvideo"
                if item["id"].count("_")==1:
                    idx = int(item["id"].split("_")[0])
                    model_name = item["id"].split("_")[1]
                else:
                    idx = int(item["id"].split("_")[2])
                    model_name = item["id"].split("_")[3]
                    
                if idx < last_idx:
                    source_idx += 1
                last_idx = idx
                if source_idx != target_source_idx:
                    continue
                idx = f"{source_list[source_idx]}-{idx}"
                
                
                
                if idx not in ref_data_map:
                    continue
                if idx not in data_map:
                    data_map[idx] = {"model_scores": {}}
                assert item['text'] == ref_data_map[idx]["prompt"], f"{item['text']} != {ref_data_map[idx]['prompt']}"
                scores = ast.literal_eval(item['ans'])
                data_map[idx]["model_scores"][model_name] = {
                    aspects[i]: scores[i] for i in range(len(aspects))
                }
                
            
            # add preference of existing models
            print("Total pairs: ", total_pairs) 
            for idx in data_map.keys():
                model_with_scores = list(data_map[idx]["model_scores"].keys())
                data_map[idx]['preference'] = {
                    left_model_name: {
                        right_model_name: ref_data_map[idx]["preference"][left_model_name][right_model_name]
                        for right_model_name in model_with_scores
                        if right_model_name in ref_data_map[idx]["preference"][left_model_name]
                    }
                    for left_model_name in model_with_scores 
                    if left_model_name in model_with_scores
                }
            with open(f"{result_dir}/data_map.json", 'w') as f:
                json.dump(data_map, f, indent=4)
            cur_total_pairs = count_total_pairs(data_map)
            if total_pairs != cur_total_pairs:
                print(f"Missing pairs: {total_pairs - cur_total_pairs}")
            
            acc = 0
            aspect_accs = {aspect: 0 for aspect in aspects}
            for idx in data_map.keys():
                
                for left_model_name in data_map[idx]["preference"]:
                    for right_model_name in data_map[idx]["preference"][left_model_name]:
                        left_score = np.mean(list(data_map[idx]["model_scores"][left_model_name].values()))
                        right_score = np.mean(list(data_map[idx]["model_scores"][right_model_name].values()))
                        left_preference = data_map[idx]["preference"][left_model_name][right_model_name]
                        right_preference = 1 - left_preference
                        if get_pairwise_acc(left_score, right_score, left_preference, right_preference):
                            acc += 1
                            for aspect in aspects:
                                left_score = data_map[idx]["model_scores"][left_model_name][aspect]
                                right_score = data_map[idx]["model_scores"][right_model_name][aspect]
                                if get_pairwise_acc(left_score, right_score, left_preference, right_preference):
                                    aspect_accs[aspect] += 1
            
            mean_acc = round(acc/cur_total_pairs, 4) if cur_total_pairs != 0 else 0
            print(f"File: {res_file}")
            print(f"acc: {mean_acc}")
            for aspect in aspects:
                aspect_accs[aspect] = round(aspect_accs[aspect]/cur_total_pairs, 4) if cur_total_pairs != 0 else 0
                print(f"{aspect}: {aspect_accs[aspect]}")
                
            all_model_acc[file_model_name] = {
                "acc": mean_acc,
                "aspect_accs": {aspect: aspect_accs[aspect] for aspect in aspects},
                "total_examples": cur_total_pairs,
            }
        
        
        # sort all_model_acc by acc
        all_model_acc = {k: v for k, v in sorted(all_model_acc.items(), key=lambda item: item[1]["acc"], reverse=True)}
        pt_table = pt.PrettyTable()
        abb_aspects = {
            "visual quality": "VQ",
            "temporal consistency": "TC",
            "dynamic degree": "DD",
            "text-to-video alignment": "TVA",
            "factual consistency": "FC"
        }
        pt_table.title = f"Pairwise Accuracy on vbench ({source_name}) using Mean of 5 aspects or each single aspect score" + (" (Excluding tie)" if exclude_tie else "")
        pt_table.field_names = ["Model", "Mean Acc"] + [abb_aspects[aspect] for aspect in aspects] + ["Total Examples"]
        # random
        for model_name, model_acc in all_model_acc.items():
            # model_name = "\n".join(wrap(model_name, 50))
            pt_table.add_row([model_name] + [model_acc["acc"]] + [model_acc["aspect_accs"][aspect] for aspect in aspects] + [model_acc["total_examples"]])
        print(pt_table)
        os.makedirs(csv_dir, exist_ok=True)
        with open(os.path.join(csv_dir, f"vbench_pairwise_acc_{source_name}_{'exclude_tie' if exclude_tie else 'include_tie'}.csv"), "w") as f:
            f.write(pt_table.get_csv_string())
        
    

if __name__ == "__main__":
    fire.Fire(main)
    
    
