import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from scipy.special import softmax
from timm.utils import accuracy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "VideoMAEv2"))

import utils


@torch.no_grad()
def final_test(data_loader, model, device, file, meta_info=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # convert to float32 type
        logits = output.float()
        # apply softmax function
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        for i in range(output.size(0)):
            string = f"{ids[i]} {output.data[i].cpu().numpy().tolist()!s} {probabilities.data[i].cpu().numpy().tolist()!s} {int(target[i].cpu().numpy())!s} {int(chunk_nb[i].cpu().numpy())!s} {int(split_nb[i].cpu().numpy())!s}\n"
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        if not os.path.exists(file):
            os.mknod(file)
        with open(file, "w") as f:
            f.write(f"{acc1}, {acc5}\n")
            for line in final_result:
                f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        f"* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}"
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def final_merge(eval_path, num_tasks, meta_info_path, method="prob"):
    assert method in ["prob", "score"]
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + ".txt")
        with open(file) as stream:
            lines = stream.readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split("[")[0]
            label = line.split("]")[2].split(" ")[1]
            chunk_nb = line.split("]")[2].split(" ")[2]
            split_nb = line.split("]")[2].split(" ")[3]
            data = np.fromstring(line.split("[")[1].split("]")[0], dtype=float, sep=",")
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            if method == "prob":
                dict_feats[name].append(softmax(data))
            else:
                dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    p = Pool(64)
    ans = p.map(compute_video, input_lst)

    with open(meta_info_path) as f:
        meta_info = json.load(f)

    # Iterate over the results in ans to update the corresponding meta_info entries
    for row in ans:
        video_id, prob, _, _, _, _ = row
        video_id = video_id.rstrip()

        for item in meta_info:
            if item["filepath"] == video_id:
                print(f"Updating {video_id}")
                prob_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                total_score = np.sum(prob * prob_weights)
                item["commonsense_adherence_score"] = total_score
                break

    return meta_info


def compute_video(lst):
    i, video_id, data, label = lst
    feat = list(data)
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [video_id, feat, pred, top1, top5, int(label)]
