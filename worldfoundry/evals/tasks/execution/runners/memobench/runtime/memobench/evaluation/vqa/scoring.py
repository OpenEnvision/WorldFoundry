import csv
import json

if __name__ == "__main__":
    inputs = []
    with open("failure-cases.csv", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            inputs.append(row)
    print(inputs)

    outputs = []
    for input_config in inputs:
        if input_config["Finished"] != "Done":
            continue

        scene = input_config["scene"]
        video_id = input_config["video_id"]

        score = []
        with open(f"score/Lingbot/{scene}-{video_id}.csv", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                score.append(row)
        score = json.loads(score[0]["score"])

        output = {}
        output["scene"] = scene
        output["video_id"] = video_id
        for dimension in score:
            output[dimension] = score[dimension]

        outputs.append(output)

    fieldnames = output.keys()
    with open("score/Lingbot/overall.csv", "w", newline="", encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(outputs)
