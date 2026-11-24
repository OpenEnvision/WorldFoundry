import argparse
import json
import os
import re
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Merge JSON files with total average scores.")
    parser.add_argument(
        "--folder_path", type=str, default="results/all", help="The path to the folder containing the JSON files."
    )
    parser.add_argument("--output_path", type=str, default="results", help="The filename for the merged results.")
    return parser.parse_args()


def main(folder_path, output_path):
    results = defaultdict(list)
    prefix_pattern = re.compile(r"^(.*)_.*_MTScore\.json$")

    for filename in os.listdir(folder_path):
        match = prefix_pattern.match(filename)
        if match:
            prefix = match.group(1)
            with open(os.path.join(folder_path, filename)) as file:
                data = json.load(file)
                results[prefix].append(data["average_metamorphic_score"])

    final_results = {}
    for prefix, scores in results.items():
        average_score = sum(scores) / len(scores)
        final_results[prefix] = {"Average_MTScore": average_score}

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    output_json = os.path.join(output_path, "merge_MTScore.json")
    with open(output_json, "w") as output_file:
        json.dump(final_results, output_file, indent=4)

    print(f"Results have been merged and saved to {output_json}")


if __name__ == "__main__":
    args = parse_args()
    main(args.folder_path, args.output_path)
