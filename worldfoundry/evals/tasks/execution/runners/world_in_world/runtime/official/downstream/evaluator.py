import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from functools import partial, wraps

import numpy as np
from downstream.downstream_datasets import AEQADataset, ARDataset
from downstream.process_IGnav_dataset.pickle_dataset import load_igdataset_from_zip
from downstream.utils.saver import Saver
from openeqa.evaluation.llm_match import get_llm_match_score
from tabulate import tabulate
from tqdm import tqdm

# from openeqa.data import results


def compute_ar_eval_metrics(gts, preds, traj_lens):
    acc = np.mean(np.array(gts) == np.array(preds))
    mean_traj_len = np.mean([t for t in traj_lens if t is not None])

    return acc, mean_traj_len


def compute_aeqa_eval_metrics(scores, traj_lens, demo_lens):
    assert None not in scores, [i for i, score in enumerate(scores) if score is None]
    mean_score = np.mean([(sigma - 1) * 25 for sigma in scores])
    mean_traj_len = np.mean([p for p in traj_lens if p is not None])
    mean_efficiency = np.mean(
        [
            (sigma - 1) * 25 * demo_len / max(p, demo_len)
            for sigma, p, demo_len in zip(scores, traj_lens, demo_lens, strict=False)
            if p is not None
        ]
    )

    return mean_score, mean_traj_len, mean_efficiency


def compute_vln_eval_metrics(s, p, l):  # noqa: E741 - Public parameter name; retain the signature.
    """
    s: is_success
    p: traj_dist
    l: demo_dist
    """
    s = np.array(s)
    p = np.array(p)
    l = np.array(l)  # noqa: E741 - Public parameter name; retain the signature.
    spl = s * l / np.maximum(p, l)
    return s.mean(), spl.mean()


def load_aeqa_demo_trajlens(full_demo_lens_path="downstream/others/gt_path_length.json"):
    with open(full_demo_lens_path) as f:
        full_demo_lens = json.load(f)
    return full_demo_lens


def rerun(func, result_path):
    if osp.exists(result_path):

        @wraps(func)
        def load_result(*args, **kwargs):
            with open(result_path, encoding="utf-8") as f:
                return json.load(f)

        return load_result
    else:
        return func


class Evaluator:
    def __init__(
        self,
        scorer_args,
        saver_args,
        blind_result_path="",
        dataset_args=None,
        eval_blind=False,
        only_check_exist=False,
    ):
        """
        Input:
            scorer_args: dict w/ keys:
                - openai_key
                - openai_model
                - openai_base_url
            saver_args: dict w/ keys:
                - parallel_ith
                - parallel_total
                - exp_id
                - task
        """
        assert saver_args["task"] in ["AEQA", "AR", "VLN", "ObjNav", "IGNav"]
        # assert "8000" in scorer_args["openai_base_url"]  # * vllm
        if saver_args["task"] == "AR":
            self.full_demo_lens = {}
            self.dataset = ARDataset(subset=dataset_args)
        elif saver_args["task"] == "AEQA":
            self.dataset = AEQADataset(
                subset_size=int(dataset_args),
                saved_episodes_path="data/WIW_datasets/eval_datasets/AEQA/episodes_AEQA.json.gz",
            )
            self.scorer = partial(get_llm_match_score, **scorer_args)
            self.full_demo_lens = load_aeqa_demo_trajlens()
        elif saver_args["task"] == "IGNav":
            self.dataset = load_igdataset_from_zip(
                "data/WIW_datasets/eval_datasets/IGNav/igdataset_goal_imgs.zip",
                "data/WIW_datasets/eval_datasets/IGNav/episodes_IGNav.json.gz",
            )
        else:
            raise ValueError(f"Task {saver_args['task']} not supported")
        self.task = saver_args["task"]
        self.saver = Saver(**saver_args)
        # * result
        self.result_path = self.saver.get_metric_path(auto_create_dir=False)
        self.result = defaultdict(lambda: defaultdict(dict))
        self.result_w_blind = defaultdict(lambda: defaultdict(dict))

        # if blind_result_path is not empty, we use blind LLMs to get score_with_blind
        if blind_result_path and saver_args["task"] == "AEQA":
            self.blind_result = json.load(open(blind_result_path, encoding="utf-8"))  # noqa: SIM115 - The handle is read once inline as upstream.
        else:
            self.blind_result = None
        self.eval_blind = eval_blind
        self.only_check_exist = only_check_exist

    def evaluate_all(self):
        for datum in tqdm(self.dataset, desc=self.saver.exp_id.join("[]")):
            self.evaluate_one(datum, only_check_exist=self.only_check_exist)
            # self.result_path = "metrics.jsonl"

        if self.task == "AEQA":
            # result, result_w_blind = self.result, self.result_w_blind
            # !! could easily change incorrectly formatted result manually
            # !! then rerun this script
            result = self.calculate_summary_aeqa(self.result)
            result_w_blind = self.calculate_summary_aeqa(self.result_w_blind)
            with open(self.result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            with open(self.result_path.replace(".json", "_blind.json"), "w") as f:
                json.dump(result_w_blind, f, indent=2, ensure_ascii=False)

            # print necessary info:
            headers = ["result type", *list(result["summary"].keys())]
            # Prepend a label to each row for the result type
            row = ["Result", *list(result["summary"].values())]
            row_ = ["Result w/ blind", *list(result_w_blind["summary"].values())]

            print(tabulate([row, row_], headers=headers, tablefmt="github"))

        elif self.task in ["VLN", "ObjNav", "IGNav"]:
            if self.task in ["VLN", "ObjNav"]:
                result = self.calculate_summary_vln(self.result)
            elif self.task == "IGNav":
                result = self.calculate_summary_ignav(self.result)

            with open(self.result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            headers = ["Total samples", "Success rate", "SPL", "Mean traj len"]
            # fmt: off
            row = [
                [result["summary"]["total_size"]],
                [f"{result['summary']['sr']:.2%}"],
                [f"{result['summary']['spl']:.2%}"],
                [f"{result['summary']['mean_traj_len']:.1f}"],
            ]
            # fmt: on
            print(tabulate([row], headers=headers, tablefmt="github"))

        elif self.task == "AR":
            result = self.calculate_summary_ar(
                self.result,
            )
            with open(self.result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            headers = ["Metric", "Value"]
            rows = [
                ["Total samples", result["summary"]["total_size"]],
                ["Correct", result["summary"]["correct"]],
                ["Accuracy", f"{result['summary']['accuracy']:.4f}"],
                ["Mean traj len", f"{result['summary']['mean_traj_len']:.3f}"],
            ]
            print(tabulate(rows, headers=headers, tablefmt="github"))

        else:
            raise ValueError(f"Task {self.task} not supported")

    def evaluate_one(self, datum, only_check_exist=False):
        one_metric_path = self.saver.get_metric_path(datum, auto_create_dir=False)
        if not osp.exists(one_metric_path):
            print(f"Metric File dir not found: {os.path.dirname(one_metric_path)}")
            return
        if only_check_exist:
            return

        with open(one_metric_path) as f:
            content = json.load(f)

        if self.task in ["AEQA", "AR"]:
            pred = content["pred"]
            traj_len = content["traj_len"]
            traj_dist = content.get("traj_dist", None)
            datum["pred"] = pred
            if self.task == "AR":
                datum["question_id"] = one_metric_path
                datum["answer"] = datum["target_categrory"]
                self.evaluate_ar(datum, pred, traj_len)
            elif self.task == "AEQA":
                self.evaluate_aeqa(datum, pred, traj_len, traj_dist)
            else:
                raise ValueError(f"Task {self.task} not supported")
        elif self.task in ["VLN", "ObjNav", "IGNav"]:
            s = content["is_success"]
            # p = content["traj_dist"]
            # l = content["demo_dist"]
            # traj_len = content["traj_len"]
            # _, spl = compute_vln_eval_metrics(s=[s], p=[p], l=[l])
            # * NOTE: datum["question_id"] = one_metric_path
            self.result["details"][one_metric_path]["sr"] = s
            self.result["details"][one_metric_path]["traj_dist"] = content.get("traj_dist", None)
            self.result["details"][one_metric_path]["demo_dist"] = content.get("demo_dist", None)
            self.result["details"][one_metric_path]["demo_len"] = content.get("demo_len", None)
            self.result["details"][one_metric_path]["traj_len"] = content["traj_len"]
        else:
            raise ValueError(f"Task {self.task} not supported")

    def evaluate_aeqa(self, datum, pred, traj_len, traj_dist):
        if datum["question_id"] not in self.blind_result:
            return
        score, score_with_blind = self.get_score(datum)
        self.set_one_result(self.result, datum, pred, traj_len, score, traj_dist)
        self.set_one_result(self.result_w_blind, datum, datum["pred"], traj_len, score_with_blind, traj_dist)

    def evaluate_ar(self, datum, pred, traj_len):
        score = int(datum["target_categrory"] == pred)
        self.set_one_result(self.result, datum, pred, traj_len, score, -1)

    def set_one_result(self, result, datum, pred, traj_len, score, traj_dist):
        if self.task in ["AEQA", "AR"]:
            result["details"][datum["question_id"]]["score"] = score
            result["details"][datum["question_id"]]["traj_len"] = traj_len
            result["details"][datum["question_id"]]["traj_dist"] = traj_dist
            result["details"][datum["question_id"]]["demo_len"] = self.full_demo_lens.get(datum["question_id"], None)
            result["details"][datum["question_id"]]["QA"] = {
                # "question": datum["question"],
                "pred": pred,
                "gt": datum["answer"],
            }
            if self.task == "AEQA":
                result["details"][datum["question_id"]]["QA"]["question"] = datum["question"]
        elif self.task == "VLN":
            raise NotImplementedError()
        else:
            raise ValueError(f"Task {self.task} not supported")

    def get_score(self, datum):
        if datum["pred"] == "No Answer" or self.eval_blind:
            if self.blind_result:
                score = 1
                datum["pred"] = self.blind_result[datum["question_id"]]
                score_with_blind = self.scorer(
                    question=datum["question"],
                    answer=datum["answer"],
                    prediction=datum["pred"],
                    extra_answers=datum.get("extra_answers"),
                )
            else:
                # when no answer, set score = 1
                score, score_with_blind = 1, 1
        else:
            score = self.scorer(
                question=datum["question"],
                answer=datum["answer"],
                prediction=datum["pred"],
                extra_answers=datum.get("extra_answers"),
            )
            score_with_blind = score
        # change subtrees/open-eqa/openeqa/evaluation/llm_match.py
        # to return error output
        return score, score_with_blind

    def calculate_summary_aeqa(self, result_var):
        scores = [info["score"] for info in result_var["details"].values()]
        valid_scores = [score for score in scores if not isinstance(score, str)]
        demo_dist = [info["demo_len"] for info in result_var["details"].values()]
        traj_dist = [info["traj_dist"] for info in result_var["details"].values()]
        (
            mean_score,
            mean_traj_len,
            mean_efficiency,
        ) = compute_aeqa_eval_metrics(
            scores=scores,
            traj_lens=traj_dist,
            demo_lens=demo_dist,
        )

        result_var["summary"] = {
            "total_size": len(scores),
            "valid_size": len(valid_scores),
            "valid_score": sum(valid_scores) / len(valid_scores),
            "mean_score": mean_score,
            "mean_traj_len": mean_traj_len,
            "mean_efficiency": mean_efficiency,
        }
        return result_var

    def calculate_summary_ar(self, result_var):
        """
        Build a concise summary for the AR task.
            - accuracy  =  (#correct) / (#samples)
            - mean_traj =  mean over finite trajectory lengths
        """
        # fmt: off
        gts       = [info["QA"]["gt"]   for info in result_var["details"].values()]
        preds     = [info["QA"]["pred"] for info in result_var["details"].values()]
        traj_lens = [info["traj_len"]   for info in result_var["details"].values()]
        # fmt: on
        easy_ep = []
        for path, info in result_var["details"].items():
            if info["score"] == 1 and info["traj_len"] == 0:
                easy_ep.append(path)
        # save easy episodes as a .txt file under the result path
        with open(osp.join(osp.dirname(self.result_path), "easy_episodes.txt"), "w") as f:
            for path in easy_ep:
                ep_idx = int(path.split("/")[-2].replace("E", ""))
                scene_id = path.split("/")[-3]
                f.write(f"{scene_id} {ep_idx}\n")

        acc, mean_traj_len = compute_ar_eval_metrics(gts, preds, traj_lens)

        result_var["summary"] = {
            "total_size": len(gts),
            "correct": sum(int(g == p) for g, p in zip(gts, preds, strict=False)),
            "accuracy": acc,
            "mean_traj_len": mean_traj_len,
        }
        return result_var

    def calculate_summary_vln(self, result_var):
        # fmt: off
        s_list = [info["sr"]        for info in result_var["details"].values()]
        p_list = [info["traj_dist"] for info in result_var["details"].values()]
        l_list = [info["demo_dist"] for info in result_var["details"].values()]
        traj_lens = [info["traj_len"] for info in result_var["details"].values()]
        # fmt: on

        sr, spl = compute_vln_eval_metrics(s_list, p_list, l_list)
        mean_traj_len = np.mean(traj_lens)

        result_var["summary"] = {
            "total_size": len(s_list),
            "sr": sr,
            "spl": spl,
            "mean_traj_len": mean_traj_len,
        }
        return result_var

    def calculate_summary_ignav(self, result_var):
        # fmt: off
        s_list = [info["sr"]        for info in result_var["details"].values()]
        p_list = [info["traj_len"] for info in result_var["details"].values()]
        l_list = [info["demo_len"] for info in result_var["details"].values()]
        traj_lens = [info["traj_len"] for info in result_var["details"].values()]
        # fmt: on

        sr, spl = compute_vln_eval_metrics(s_list, p_list, l_list)
        mean_traj_len = np.mean(traj_lens)

        result_var["summary"] = {
            "total_size": len(s_list),
            "sr": sr,
            "spl": spl,
            "mean_traj_len": mean_traj_len,
        }
        return result_var

    def run(self):
        self.evaluate_all()
        print(f"Result saved to {self.result_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id")
    parser.add_argument("--task", default="AEQA")  # * ["AEQA", "AR", "VLN", "ObjNav", "IGNav"]
    parser.add_argument("--openai_key", default="XXXXX")
    parser.add_argument("--openai_model", default="gpt-4o")  # or Qwen/Qwen2.5-VL-72B-Instruct-AWQ
    parser.add_argument("--openai_service", default="openai")  # or openai
    parser.add_argument("--scorer_host", type=str, default="localhost:8000")
    parser.add_argument(
        "--blind_result_path",
        type=str,
        default="subtrees/open-eqa/data/results/AEQA_blind/open-eqa-184-Qwen--Qwen2.5-VL-72B-Instruct-AWQ-1234.json",
    )
    parser.add_argument("--parallel_ith", default=None)
    parser.add_argument("--parallel_total", default=None)
    parser.add_argument("--dataset_args", type=str, default=None)
    parser.add_argument("--only_check_exist", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SERVICE_TO_BASE_URL = {
        "vllm": f"http://{args.scorer_host}/v1",
        "sglang": f"http://{args.scorer_host}/v1",
        "openai": "https://api.openai.com/v1",
    }
    BASE_URL_TO_SERVICE = {v: k for k, v in SERVICE_TO_BASE_URL.items()}

    scorer_args = {
        "openai_key": args.openai_key,
        "openai_model": args.openai_model,
        "openai_base_url": SERVICE_TO_BASE_URL[args.openai_service],
        "diff_retry_setting": False,
    }
    saver_args = {
        "parallel_ith": args.parallel_ith,
        "parallel_total": args.parallel_total,
        "exp_id": args.exp_id,
        "task": args.task,
    }

    if args.dataset_args is None:
        if args.task == "AR":
            args.dataset_args = "Hard"
        elif args.task == "AEQA":
            args.dataset_args = "184"  # * full
        elif args.task == "VLN":
            args.dataset_args = "100"  # * debug
        elif args.task == "ObjNav":
            args.dataset_args = "130"  # * debug_len
        elif args.task == "IGNav":
            args.dataset_args = "300"  # * debug_len
        else:
            raise ValueError(f"Task {args.task} not supported")

    evaluator = Evaluator(
        scorer_args=scorer_args,
        saver_args=saver_args,
        dataset_args=args.dataset_args,
        blind_result_path=args.blind_result_path,
        only_check_exist=args.only_check_exist,
        # eval_blind=True,
    )
    evaluator.run()
