import argparse
import json
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Process and format JSON data.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing the JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for the result JSON file")
    parser.add_argument("--model_names", nargs="+", default=["test"], help="Name of the models.")
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["open", "close"],
        default="open",
        help="Specify the evaluation mode: 'open' for open-source models or 'close' for closed-source models.",
    )
    return parser.parse_args()


def parse_entry(value):
    split_value = value.rsplit("\n", 1)
    brief_reasoning_statement = split_value[0].replace("Brief Reasoning Statement:", "").strip()
    score = split_value[1].replace('"Score":', "").replace("{", "").replace("}", "").strip()
    return brief_reasoning_statement, score


def main(input_dir, output_dir, model_name):
    if args.eval_type == "open":
        for part in tqdm(["1", "2", "3"], desc="Processing parts", leave=False):
            file_path = os.path.join(input_dir, f"{model_name}_{part}_metamorphic.json")
            result_file_path = os.path.join(output_dir, f"{model_name}_{part}_GPT4o-MTScore.json")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(file_path) as file:
                data = json.load(file)

            formatted_data = {}
            total_score = 0
            entry_count = 0
            keys_to_delete = []

            for key, value in data.items():
                try:
                    brief_reasoning_statement, score = parse_entry(value)
                    formatted_data[key] = {"Brief Reasoning Statement": brief_reasoning_statement, "Score": score}
                    if score:
                        total_score += int(score)
                        entry_count += 1
                except Exception:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del data[key]

            average_score = total_score / entry_count if entry_count > 0 else 0

            result_data = {"Average Score": average_score, "Formatted Data": formatted_data}

            with open(result_file_path, "w") as file:
                json.dump(result_data, file, indent=4)

            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
    elif args.eval_type == "close":
        file_path = os.path.join(input_dir, f"{model_name}_metamorphic.json")
        result_file_path = os.path.join(output_dir, f"{model_name}_1_GPT4o-MTScore.json")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(file_path) as file:
            data = json.load(file)

        formatted_data = {}
        total_score = 0
        entry_count = 0
        keys_to_delete = []

        for key, value in data.items():
            try:
                brief_reasoning_statement, score = parse_entry(value)
                formatted_data[key] = {"Brief Reasoning Statement": brief_reasoning_statement, "Score": score}
                if score:
                    total_score += int(score)
                    entry_count += 1
            except Exception:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del data[key]

        average_score = total_score / entry_count if entry_count > 0 else 0

        result_data = {"Average Score": average_score, "Formatted Data": formatted_data}

        with open(result_file_path, "w") as file:
            json.dump(result_data, file, indent=4)

        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    args = parse_args()

    for model_name in tqdm(args.model_names, desc="Processing model"):
        main(args.input_dir, args.output_dir, model_name)
