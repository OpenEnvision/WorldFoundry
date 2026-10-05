import ast
import json
import numpy as np
import scipy.stats as stats
import os
import fire

ROUND_DIGIT=4

def cal_spearman_correlation(
    result_dir: str="./benchmark/eval_results/video_feedback/",
    bench_name: str="video_feedback"
):
    for result_file in sorted(os.listdir(result_dir)):
        if not result_file.startswith("eval_"):
            continue
        # result_file example: eval_video_feedback_mantisscore.json
        model_name=result_file.split(".")[0].split("_")[-1]
        all_res=json.load(open(f"{result_dir}/{result_file}","r"))
        all_ref_scores=[ast.literal_eval(item["ref"]) for item in all_res]
        all_ans_scores=[ast.literal_eval(item["ans"]) for item in all_res]
        
        spearman_list=[]
        p_value_list=[]
        try:
            all_ref_scores=np.array(all_ref_scores)
            all_ans_scores=np.array(all_ans_scores)
            for i in range(len(all_ref_scores[0])):
                ref_list=[x[i] for x in all_ref_scores]
                ans_list=[x[i] for x in all_ans_scores]
                rho,p_value=stats.spearmanr(ref_list,ans_list)
                if not np.isnan(rho):
                    rho*=100
                    spearman_list.append(round(rho,ROUND_DIGIT))
                    p_value_list.append(p_value)
                else:
                    spearman_list.append(None)
                    p_value_list.append(None)    
        except Exception as e:
            print(e)
            spearman_list=[None for _ in range(len(all_ref_scores[0]))]
            p_value_list=[None for _ in range(len(all_ref_scores[0]))]
        print(f"metric: {result_file.split('eval_')[1].split('.')[0]}, SPCC: {spearman_list}")
        
        dirname=os.path.dirname(f"{result_dir}/{result_file}")
        spcc_file=f"{dirname}/spearman_corr_{bench_name}.json"
        
        if not os.path.exists(spcc_file):
            all_spcc=[]
        else:
            all_spcc=json.load(open(spcc_file,"r"))
        all_spcc.append({model_name:
                {
                "spearman_list":spearman_list,
                "p_value_list":p_value_list,
                }
            })
        
        with open(spcc_file,"w") as file:
            json.dump(all_spcc,file,indent=4)
        
if __name__ == "__main__":
    fire.Fire(cal_spearman_correlation)