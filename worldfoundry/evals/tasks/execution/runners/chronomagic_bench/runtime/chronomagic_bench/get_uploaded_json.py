import argparse
import json
import os
import re
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Merge JSON files with total average scores.")
    parser.add_argument(
        "--input_path", type=str, default="results/all", help="The path to the folder containing the JSON files."
    )
    parser.add_argument("--output_path", type=str, default="results", help="The filename for the merged results.")
    return parser.parse_args()


def merge_scores(input_path, output_path, patterns, score_keys, result_keys, output_filename):
    results = defaultdict(lambda: defaultdict(list))

    for filename in os.listdir(input_path):
        for pattern, score_key, result_key in zip(patterns, score_keys, result_keys, strict=False):
            prefix_pattern = re.compile(pattern)
            match = prefix_pattern.match(filename)
            if match:
                prefix = match.group(1)
                with open(os.path.join(input_path, filename)) as file:
                    data = json.load(file)
                    results[prefix][result_key].append(data[score_key])

    final_results = {}
    for prefix, score_dict in results.items():
        final_results[prefix] = {result_key: sum(scores) / len(scores) for result_key, scores in score_dict.items()}

    final_results[prefix]["UMT-FVD"] = -1
    final_results[prefix]["UMTScore"] = -1

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_json = os.path.join(output_path, output_filename)
    with open(output_json, "w") as output_file:
        json.dump(final_results, output_file, indent=4)

    print(f"Results have been merged and saved to {output_json}")


def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path

    patterns = [r"^(.*)_.*_CHScore\.json$", r"^(.*)_.*_GPT4o-MTScore\.json$", r"^(.*)_.*_MTScore\.json$"]

    score_keys = ["total_average_score", "Average Score", "average_metamorphic_score"]

    result_keys = ["Average_CHScore", "Average_GPT4o-MTScore", "Average_MTScore"]

    merge_scores(input_path, output_path, patterns, score_keys, result_keys, "ChronoMagic-Bench-Input.json")


if __name__ == "__main__":
    args = parse_args()
    main()
