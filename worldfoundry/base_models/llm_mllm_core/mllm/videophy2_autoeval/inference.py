import argparse
import re
from collections import defaultdict

import pandas as pd
import torch
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from peft import LoraConfig, get_peft_model
from template import *  # noqa: F403 - Upstream star import kept as written.
from tqdm import tqdm
from transformers.models.llama.tokenization_llama import LlamaTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--input_csv", type=str, required=True, help="csv")
parser.add_argument("--task", type=str, default="sa", choices=["sa", "pc", "rule"], help="task")
parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint")
parser.add_argument("--lora_checkpoint", default=None, type=str, help="lora trained ckpt")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_frames", type=int, default=32)
parser.add_argument("--output_csv", type=str, required=True, help="csv")

args = parser.parse_args()

generate_kwargs = {
    "do_sample": False,
    "top_k": 1,
    "temperature": 0.001,
    "max_length": 256,
}


def inference(args, model, df, processor, tokenizer):
    num_map = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
    }

    with torch.no_grad():
        for i, row in tqdm(df.iterrows()):
            videopaths = [row["videopath"]]
            if args.task == "sa":
                prompts = [PROMPT_SA.format(caption=row["caption"])]  # noqa: F405 - Name comes from the upstream star import.
            elif args.task == "pc":
                prompts = [PROMPT_PHYSICS]  # noqa: F405 - Name comes from the upstream star import.
            else:
                prompts = [PROMPT_RULE.format(rule=row["rule"])]  # noqa: F405 - Name comes from the upstream star import.
            inputs = processor(text=prompts, videos=videopaths, num_frames=args.num_frames, return_tensors="pt")
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            res = model.generate(**inputs, **generate_kwargs)
            output = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
            output_lower = output.lower().strip()

            print(output_lower)
            score = None
            for key, val in num_map.items():
                if key in output_lower:
                    score = val
                    break

            if score is None:
                # Optionally, try to extract a digit with a simple filter.
                digits = "".join([c for c in output_lower if c.isdigit()])
                score = int(digits) if digits and int(digits) in num_map.values() else 0
                print(f"Warning: Could not parse output '{output}'. Defaulting to {score}.")

            # Set the parsed score as an integer in the dataframe.
            df.at[i, "score"] = score
    return df


def modify_keys(state_dict):
    new_state_dict = defaultdict()

    pattern = re.compile(r".*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj).weight")

    for key, value in state_dict.items():
        if pattern.match(key):
            key = key.split(".")
            key.insert(-1, "base_layer")
            key = ".".join(key)
        new_state_dict[key] = value

    return new_state_dict


def main():
    checkpoint = args.checkpoint

    # Processors
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    df = pd.read_csv(args.input_csv)
    # df = df.iloc[:20]

    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    # Instantiate model
    model = MplugOwlForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    print("Model Loaded")
    model.eval()

    lora_checkpoint = args.lora_checkpoint
    if lora_checkpoint:
        peft_config = LoraConfig(
            target_modules=r".*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)",
            inference_mode=True,
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        with open(lora_checkpoint, "rb") as f:
            ckpt = torch.load(f, map_location=torch.device("cpu"))
        try:
            model.load_state_dict(ckpt)
        except BaseException:
            ckpt = modify_keys(ckpt)
            model.load_state_dict(ckpt)
        print("LOADED")
    model = model.to("cuda").to(torch.bfloat16)

    out = inference(args, model, df, processor, tokenizer)
    out.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()

"""
    CUDA_VISIBLE_DEVICES=0 python inference.py --input_csv /local/hbansal/videophy2/human_expts/cogvideox_5b_3x_videophy2_hard_pre_human_annotation_w_bad_sa_pc_eval.csv --checkpoint /local2/hbansal/videophy2/test_videophy_training/videophy_autoeval_three_models_rule_e3_lr5e-4_bs64_part2_vta_pc_rule/videophy_2_autoeval --output_csv /local/hbansal/videophy2/human_expts/cogvideox_5b_3x_videophy2_hard_pre_human_annotation_w_bad_sa_pc_eval_lr5e-4_bs64_part2_vta_pc_rule_502_rerun.csv
"""
