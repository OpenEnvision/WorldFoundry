import argparse
import csv
import json
from statistics import mean


def calculate_averages(json_file, output_csv):
    with open(json_file) as f:
        data = json.load(f)

    scores = {
        "perceptible_amplitude_socre": [],
        "object_integrity_score": [],
        "temporal_coherence_score": [],
        "motion_smoothness_score": [],
        "commonsense_adherence_score": [],
    }

    for item in data:
        for key in scores:
            if key in item:
                scores[key].append(item[key])

    averages = {key: mean(values) for key, values in scores.items() if values}

    total_score = mean(averages.values()) if averages else 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Average Score"])
        for key, value in averages.items():
            writer.writerow([key, value * 100])
        writer.writerow(["Total Score", total_score * 100])

    print(f"Results have been saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Calculate average scores from JSON and save to CSV")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output CSV file")

    args = parser.parse_args()

    calculate_averages(args.input, args.output)


if __name__ == "__main__":
    main()
