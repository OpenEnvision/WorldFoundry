import ast
import os
import json
import fire
import random
import numpy as np
import prettytable as pt
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

def main(
    data_repo_name: str="TIGER-Lab/VideoFeedback-Bench",
    result_dir: str="./benchmark/eval_results/genaibench",
    exclude_tie:bool=False,
    seed: int=42,
):
    random.seed(seed)
    
    ref_data=load_dataset(data_repo_name,name="genaibench",split="test")

    ref_data_map = {}
    for idx,item in enumerate(ref_data):
        # id example: 0000_left
        idx = str(int(item["id"].split("_")[0]))
        left_or_right = item["id"].split("_")[1]
        if idx not in ref_data_map:
            ref_data_map[idx] = {}
        preference = item['preference']
        ref_data_map[idx][left_or_right+"_preference"] = preference
        
    if exclude_tie:
        ref_data_map = {idx: item for idx, item in ref_data_map.items() if item["left_preference"] != item["right_preference"]}
        
    all_model_acc = {}
    
    # add random as a model
    random_acc = 0
    for idx, item in ref_data_map.items():
        left_preference = item["left_preference"]
        right_preference = item["right_preference"]
        left_score = random.random() * 4
        right_score = random.random() * 4
        if get_pairwise_acc(left_score, right_score, left_preference, right_preference):
            random_acc += 1
    random_acc = round(random_acc/len(ref_data_map), 4)
    all_model_acc["Random"] = {
        "acc": random_acc,
        "aspecct_accs": {aspect: random_acc for aspect in aspects},
        "total_examples": len(ref_data_map),
    }
        
    for res_file in sorted(os.listdir(result_dir)):
        if not res_file.startswith("eval_"):
            continue
        print(f"Processing {res_file}")
        model_name = res_file.split(".")[0].split("_")[-1]
        with open(f"{result_dir}/{res_file}", 'r') as f:
            data = json.load(f)
        data_map = {}
        for item in data:
            # "0001_left",
            idx = str(int(item["id"].split("_")[0]))
            left_or_right = item["id"].split("_")[1]
            if idx not in ref_data_map:
                continue
            if idx not in data_map:
                data_map[idx] = {}
                
            scores = ast.literal_eval(item['ans'])
            data_map[idx][left_or_right] = {
                aspects[i]: scores[i] for i in range(len(aspects))
            }
            data_map[idx][left_or_right+"_preference"] = ref_data_map[idx][left_or_right+"_preference"]

        # assert all have "left" and "right"
        print("Total examples: ", len(data_map))
        to_remove_idx = []
        for idx in data_map.keys():
            if "left" not in data_map[idx] or "right" not in data_map[idx]:
                to_remove_idx.append(idx)
        for idx in to_remove_idx:
            data_map.pop(idx)
        print("Removed examples: ", len(to_remove_idx), "due to missing left or right")
        
        random_acc = 0
        acc = 0
        aspecct_accs = {aspect: 0 for aspect in aspects}
        for idx in data_map.keys():
            left = data_map[idx]["left"]
            right = data_map[idx]["right"]
            left_preference = data_map[idx]["left_preference"]
            right_preference = data_map[idx]["right_preference"]
            
            for aspect in aspects:
                left_score = left[aspect]
                right_score = right[aspect]
                if get_pairwise_acc(left_score, right_score, left_preference, right_preference):
                    aspecct_accs[aspect] += 1
            left_score = np.mean([left[aspect] for aspect in aspects])
            right_score = np.mean([right[aspect] for aspect in aspects])
            if get_pairwise_acc(left_score, right_score, left_preference, right_preference):
                acc += 1
                
        print(f"File: {res_file}")
        print(f"acc: {acc/len(data_map)}")
        for aspect in aspects:
            print(f"{aspect}: {aspecct_accs[aspect]/len(data_map)}")
        
        all_model_acc[model_name] = {
            "acc": round(acc/len(data_map), 4),
            "aspecct_accs": {aspect: round(aspecct_accs[aspect]/len(data_map), 4) for aspect in aspects},
            "total_examples": len(data_map),
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
    pt_table.title = "Pairwise Accuracy on Genaibench using Mean of 5 aspects or each single aspect score" + (" (Excluding tie)" if exclude_tie else "")
    pt_table.field_names = ["Model", "Mean Acc"] + [abb_aspects[aspect] for aspect in aspects] + ["Total Examples"]
    # random
    for model_name, model_acc in all_model_acc.items():
        # model_name = "\n".join(wrap(model_name, 50))
        pt_table.add_row([model_name] + [model_acc["acc"]] + [model_acc["aspecct_accs"][aspect] for aspect in aspects] + [model_acc["total_examples"]])
    print(pt_table)
    
    
if __name__ == "__main__":
    fire.Fire(main)
    
    
