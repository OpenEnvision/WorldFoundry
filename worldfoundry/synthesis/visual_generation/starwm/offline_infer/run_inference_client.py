import json
import argparse
import requests
import re
import os
from tqdm import tqdm
import concurrent.futures

def process_prompt(original_content, mode):

    new_prompt = original_content.strip()
    if new_prompt.endswith("/no_think"):
        new_prompt = new_prompt[:-len("/no_think")].strip()
    if new_prompt.endswith("/think"):
        new_prompt = new_prompt[:-len("/think")].strip()
        
    # Apply mode
    if mode == "nothink":
        new_prompt += "\n\n/no_think"
    elif mode == "think":
        # For think mode, we append /think to force thinking if supported or intended
        new_prompt += "\n\n/think"
        
    return new_prompt

def call_vllm(prompt, api_base, api_key, model_id, mode):
    url = f"{api_base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6, # Zero-shot usually implies greedy or low temp for reproducibility
        "max_tokens":5000  if mode == "nothink" else 15000, # Adjust as needed
        "top_p": 0.95,
        "top_k": 20,
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error calling VLLM: {e}")
        return ""

def parse_output(text):
    # Look for <think>...</think>
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = think_pattern.search(text)
    
    if match:
        think_content = match.group(1).strip()
        # Output is everything after </think>
        output_content = text[match.end():].strip()
        # Also could be everything before <think> + everything after? 
        # Usually think is at start. We assume standard format.
    else:
        think_content = ""
        output_content = text.strip()
        
    return think_content, output_content

def process_single_item(item, args):
    user_msg = next((m for m in item['messages'] if m['role'] == 'user'), None)
    assistant_msg = next((m for m in item['messages'] if m['role'] == 'assistant'), None)
    
    if not user_msg:
        return None
        
    original_prompt = user_msg['content']
    label = assistant_msg['content'] if assistant_msg else ""
    
    # Construct prompt
    final_prompt = process_prompt(original_content=original_prompt, mode=args.mode)

    # Call API
    full_response = call_vllm(final_prompt, args.api_base, args.api_key, args.model_id, mode = args.mode)
    
    # Parse
    think_content, output_content = parse_output(full_response)
    
    return {
        "input": final_prompt,
        "think": think_content,
        "output": output_content,
        "labels": label
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mode", choices=["think", "nothink"], required=True)
    parser.add_argument("--api_base", type=str, default="http://localhost:12000")
    parser.add_argument("--api_key", type=str, default="sk-11223344")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent workers")
    args = parser.parse_args()
    
    if args.input_file.endswith(".json"):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif args.input_file.endswith(".jsonl"):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        
    flag = True
    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Create a list of futures
            futures = [executor.submit(process_single_item, item, args) for item in data]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    if flag:
                        print(f"input: \n{result['input']}\n")
                        print(f"think: \n{result['think']}\n")
                        print(f"output: \n{result['output']}\n")
                        flag = False

                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
            
    print(f"Done. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
