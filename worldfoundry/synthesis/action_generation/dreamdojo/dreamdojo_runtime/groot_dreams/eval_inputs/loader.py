import torch
import random
import numpy as np
from pathlib import Path
import yaml

from groot_dreams.eval_inputs.lerobot_sequences import WrappedLeRobotSingleDataset
from groot_dreams.groot_configs import construct_modality_config_and_transforms
from groot_dreams.eval_inputs.dataset_mano import MANODataset
from groot_dreams.eval_inputs.dataset_video import VideoDataset


def is_lerobot_dataset(dataset_path: str) -> bool:
    return "gr1" in dataset_path.lower() or "g1" in dataset_path.lower() or "yam" in dataset_path.lower() or "agibot" in dataset_path.lower()


class VideoActionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        height=None,
        width=None,
        dataset_path=None,
        data_split="train",
        embodiment=None,
        agibot_pad_freq10=False,
        waist_concat=False,
        single_base_index=False,
    ):
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.height = height
        self.width = width

        config, train_transform, test_transform = construct_modality_config_and_transforms(
            num_frames=(num_frames + 1), embodiment=embodiment, agibot_pad_freq10=agibot_pad_freq10, waist_concat=waist_concat
        )  # Add an additional prefix frame as baseline to compute delta actions
        self.lerobot_dataset = WrappedLeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=config,
            transforms=train_transform if data_split == "train" else test_transform,
            embodiment_tag=embodiment,
            data_split=data_split,
            num_frames=num_frames,
            single_base_index=single_base_index,
        )
        print(f"Loaded lerobot {data_split} dataset from {self.dataset_path} with {len(self)} samples.")

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

    def __getitem__(self, data_id):
        lerobot_data = self.lerobot_dataset[data_id]

        if lerobot_data["video"].shape[1] != self.num_frames:
            print(f"Warning: Expected {self.num_frames} frames, but got {lerobot_data['video'].shape[1]} frames. Randomly sampling an item instead.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        data = {
            "video": lerobot_data["video"],
            "lam_video": lerobot_data["lam_video"],
            "action": lerobot_data["action"],
            "dataset": self.lerobot_dataset.dataset_name,
            "fps": lerobot_data["fps"],
            "num_frames": lerobot_data["num_frames"],

            "__key__": lerobot_data["__key__"],
            "padding_mask": lerobot_data["padding_mask"],
            "image_size": lerobot_data["image_size"],
            "ai_caption": lerobot_data["ai_caption"],
        }
        return data

    def __len__(self):
        return len(self.lerobot_dataset)


class MultiVideoActionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args=None,

        dataset_path=None,
        num_frames=81,
        time_division_factor=4,
        height=None,
        width=None,
        data_split="train",
        single_base_index=False,

        deterministic_uniform_sampling=False,
        dataset_mixing_weights=None,
        restrict_len=None,

        cr1_embeddings_path=None,
    ):
        if args is not None:
            dataset_path = args.dataset_path
            height = args.height
            width = args.width
            num_frames = args.num_frames
            single_base_index = args.single_base_index
            deterministic_uniform_sampling = args.deterministic_uniform_sampling
            dataset_mixing_weights = args.dataset_mixing_weights

        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.height = height
        self.width = width
        self.deterministic_uniform_sampling = deterministic_uniform_sampling
        self.dataset_mixing_weights = dataset_mixing_weights
        self.restrict_len = restrict_len

        dataset_paths = []
        if isinstance(dataset_path, str):
            for p in dataset_path.split(","):
                p = p.strip()
                dataset_paths.append(p)
        else:
            dataset_paths = dataset_path

        self.datasets = []
        for path in dataset_paths:
            path = path.strip()
            if "egodex_21" in path.lower():
                self.datasets.append(MANODataset(
                    converted_root=path,
                    eval_mode=(data_split == "test"),
                    num_frames=num_frames,
                    time_division_factor=time_division_factor,
                ))
                print(f"Created MANODataset for {path}")
            elif not is_lerobot_dataset(path.lower()):
                self.datasets.append(VideoDataset(
                    video_root=path,
                    num_frames=num_frames,
                ))
                print(f"Created VideoDataset for {path}")
            else:
                if "gr1" in path.lower():
                    embodiment = "gr1"
                elif "agibot" in path.lower():
                    embodiment = "agibot"
                elif "g1" in path.lower():
                    embodiment = "g1"
                elif "yam" in path.lower():
                    embodiment = "yam"
                else:
                    raise ValueError(f"Cannot infer embodiment from dataset path: {path}")
                self.datasets.append(VideoActionDataset(
                    dataset_path=path,
                    data_split=data_split,
                    num_frames=num_frames,
                    time_division_factor=time_division_factor,
                    height=height,
                    width=width,
                    embodiment=embodiment,
                    agibot_pad_freq10=False,
                    waist_concat=False,
                    single_base_index=single_base_index,
                ))
                print(f"Created VideoActionDataset for {path}")

        if self.dataset_mixing_weights is not None:
            # self.dataset_mixing_weights = [float(w) for w in self.dataset_mixing_weights]
            # assert len(self.datasets) == len(self.dataset_mixing_weights), f"Number of datasets {len(self.datasets)} does not match number of mixing weights {len(self.dataset_mixing_weights)}."
            # # assert sum(self.dataset_mixing_weights) == 1.0, f"Dataset mixing weights should sum to 1.0, got {sum(self.dataset_mixing_weights)}."
            # self.dataset_mixing_weights = np.array(self.dataset_mixing_weights) / sum(self.dataset_mixing_weights)
            # self.dataset_shuffle_indexes = [torch.randperm(len(d)).tolist() for d in self.datasets]
            # self.dataset_iter_indexes = [0 for _ in self.datasets]
            self.dataset_mixing_weights = [float(w) for w in self.dataset_mixing_weights]
            total_prob = sum(self.dataset_mixing_weights)
            self.sample_probs = [x / total_prob for x in self.dataset_mixing_weights]
        self.t5_text_embeddings, self.t5_text_mask = None, None
        if cr1_embeddings_path is not None:
            self.t5_text_embeddings = torch.load(cr1_embeddings_path, map_location="cpu")[0]
            self.t5_text_mask = torch.ones(self.t5_text_embeddings.shape[0])
    
    def __getitem__(self, data_id):
        if self.deterministic_uniform_sampling:
            min_len = min([len(d) for d in self.datasets])
            if self.restrict_len is not None:
                assert min_len >= self.restrict_len / len(self.datasets), f"restrict_len {self.restrict_len} with {len(self.datasets)} datasets is too large for smallest dataset length {min_len}."
                min_len = self.restrict_len / len(self.datasets)
            dataset = self.datasets[int(data_id / min_len)]
            data_id = int((data_id % min_len) / min_len * len(dataset))
            ret = dataset[data_id]
        elif self.dataset_mixing_weights is not None:
            # # Stochastic dataset sampling with per-dataset item iteration (loosely based on tf.data.Dataset.sample_from_datasets)
            # dataset_idx = int(np.random.choice(len(self.datasets), p=self.dataset_mixing_weights))
            # dataset = self.datasets[dataset_idx]
            # data_id = self.dataset_shuffle_indexes[dataset_idx][self.dataset_iter_indexes[dataset_idx]]
            # self.dataset_iter_indexes[dataset_idx] += 1
            # if self.dataset_iter_indexes[dataset_idx] == len(dataset):
            #     self.dataset_iter_indexes[dataset_idx] = 0
            #     self.dataset_shuffle_indexes[dataset_idx] = torch.randperm(len(dataset)).tolist()
            subset = random.choices(self.datasets, self.sample_probs)[0]
            sample_idx = random.randint(0, len(subset) - 1)
            ret = subset[sample_idx]
        else:
            if self.restrict_len is not None:
                full_len = sum([len(d) for d in self.datasets])
                data_id = int(data_id / self.restrict_len * full_len)
            for dataset in self.datasets:
                if data_id < len(dataset):
                    break
                data_id -= len(dataset)
            ret = dataset[data_id]
        if self.t5_text_embeddings is not None and self.t5_text_mask is not None:
            ret["t5_text_embeddings"] = self.t5_text_embeddings
            ret["t5_text_mask"] = self.t5_text_mask
        return ret
    
    def __len__(self):
        if self.restrict_len:
            return self.restrict_len
        elif self.deterministic_uniform_sampling:
            return min([len(d) for d in self.datasets]) * len(self.datasets)
        else:
            return sum([len(d) for d in self.datasets])


def get_data_path(embodiment):
    yaml_path = Path(__file__).parent.parent / "configs" / f"2b_480_640_{embodiment}.yaml"
    if not yaml_path.exists():
        if embodiment == "agibot_fruit":
            return "datasets/agibot_fruit", None
        else:
            raise ValueError(f"Unknown embodiment: {embodiment}")
    
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    dataset_path = config["dataloader_train"]["dataset"]["dataset_path"]
    dataset_mixing_weights = None
    if "dataset_mixing_weights" in config["dataloader_train"]["dataset"]:
        dataset_mixing_weights = config["dataloader_train"]["dataset"]["dataset_mixing_weights"]
    return dataset_path, dataset_mixing_weights
