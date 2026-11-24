import gzip
import json
import os
import os.path as osp
import pickle
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Subset

BACKGROUND_CLASS = ["wall", "floor", "ceiling", "carpet", "door", "rug", "bath mat"]


def get_indoor_classes(type):
    if type == "scannet200":
        classes_file_path = Path("downstream/others/scannet200_classes.txt")
        with open(classes_file_path) as f:
            all_lines = [cls.strip() for cls in f.readlines()]
            all_classes = list(set(all_lines))
            all_classes = [cls for cls in all_classes if cls != "unknown"]
            print(f"Loaded {len(all_classes)} classes from {classes_file_path}")
    else:
        raise ValueError(f"Unknown type: {type}")
    return all_classes


class TaskDataset(Dataset):
    """
    If you need additional meta-data just keep them in each dictionary; the
    iterator yields whatever you store in `self.data`.
    """

    def __len__(self):
        return len(self.data)

    def filter_by_lambda(self, filter_lambda):
        filtered_indices = [i for i, x in enumerate(self) if filter_lambda(x)]
        return Subset(self, filtered_indices)


class ARDataset(TaskDataset):
    """datum_id = scene_id + episode_id"""

    OBJECT_SET = (
        "appliances",
        "bathtub",
        "bed",
        "cabinet",
        "chair",
        "chest_of_drawers",
        "clothes",
        "counter",
        "curtain",
        "cushion",
        "door",
        "fireplace",
        "furniture",
        "gym_equipment",
        "mirror",
        "picture",
        "plant",
        "seating",
        "shower",
        "sink",
        "sofa",
        "stool",
        "table",
        "toilet",
        "towel",
        "tv_monitor",
        "window",
    )  # 27

    DIFFICULTYS = ("Easy", "Moderate", "Hard")

    def __init__(
        self,
        subset=None,
        debug_len=2000,
        data_dir="data/WIW_datasets/eval_datasets/AR/",  # as default
    ):
        self.data = []
        self.mp3d_dir = "data/scene_datasets/mp3d"
        exclude_scenes, exclude_ep = set(), {}
        filtered_count = 0

        # Iterate over each scene folder in the data_dir
        sub_dirs = sorted(os.listdir(data_dir))
        for scene_id in sub_dirs:
            file_path = os.path.join(data_dir, scene_id)
            # file_path = os.path.join(dir_path, "episodes.json.gz")
            if os.path.exists(file_path):
                with gzip.open(file_path, "rt") as f:
                    episodes = json.load(f).get("episodes", [])
                    for episode in episodes:
                        # Store original scene name to check against the exclude list
                        original_scene = episode["scene_id"]
                        if original_scene in exclude_scenes:
                            filtered_count += 1
                            continue
                        if original_scene in exclude_ep and episode["episode_id"] in exclude_ep[original_scene]:
                            filtered_count += 1
                            continue
                        # Update scene_id to include the full path to the .glb file
                        # episode["scene_id"] = osp.join(
                        #     self.mp3d_dir, original_scene, f"{original_scene}.glb"
                        # )
                        self.data.append(episode)

        if subset is not None:
            self.data = [x for x in self.data if x["difficulty"] == self.DIFFICULTYS.index(subset)]
        if debug_len is not None:
            self.data = self.data[:debug_len]

        # self.data = self.filter_for_demo(self.data, self.get_demo_list())

        print("ARDataset")
        print(f"Size: {len(self)}")
        print(f"Stats: {self.get_stats()}")
        print(f"Filtered out {filtered_count} due to exclusion lists.")

    def __getitem__(self, idx):
        return self.data[idx]

    def get_stats(self):
        return {
            diff_name: len(self.filter_by_lambda(lambda x: x["difficulty"] == diff_value))  # noqa: B023 - filter_by_lambda evaluates the predicate eagerly within the iteration.
            for diff_value, diff_name in enumerate(self.DIFFICULTYS)
        }

    @staticmethod
    def get_viz_name(x):
        return osp.join(
            x["scene_id"],
            "images",
            f"{x['target_categrory']}_{x['target_id']}_000***.png",
        )


class IGDataset(TaskDataset):
    """
    If you need additional meta-data just keep them in each dictionary; the
    iterator yields whatever you store in `self.data`.
    """

    def __init__(
        self,
        debug_len=None,
        inputs_json: str = "downstream/process_IGnav_dataset/inputs.json",
        hm3d_root: str = "data/scene_datasets/hm3d/val",
        exclude_ep_path: list[str] | None = "downstream/others/IGNav_exclude_unique.txt",
        min_step_to_goal: int | None = None,
        max_step_to_goal: int | None = None,
    ) -> None:
        # This method should not be called directly. See the child class implementation
        # in downstream/process_IGnav_dataset/pickle_dataset.py for the actual initialization logic.
        raise NotImplementedError(
            "IGDataset should not be initialized directly. Use IGDatasetPortable or other subclasses."
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


class AEQADataset(TaskDataset):
    """datum_id = question_id"""

    def __init__(
        self,
        data_dir="subtrees/open-eqa/data",
        subset_size=None,
        reverse_order=False,
        saved_episodes_path=None,
    ):
        self.data = []
        self.hm3d_dir = "data/scene_datasets/hm3d"
        self.reverse_order = reverse_order

        # If a saved episodes file is provided, load from it
        if saved_episodes_path is not None and osp.exists(saved_episodes_path):
            print(f"Loading AEQA episodes from {saved_episodes_path}")
            with gzip.open(saved_episodes_path, "rt") as f:
                self.data = json.load(f)
        else:
            print("No saved episodes file provided or file does not exist. Creating episodes from original data.")
            self.data = self.create_episodes_from_aeqa_origin(data_dir, subset_size)

        print("AEQADataset")
        print(f"Size: {len(self.data)}")

    def create_episodes_from_aeqa_origin(self, data_dir, subset_size):
        assert subset_size in [41, 54, 184, 557]

        # * load qa data
        self.qa_data_path = osp.join(data_dir, f"open-eqa-{subset_size}.json")
        with open(self.qa_data_path) as f:
            self.qa_data = json.load(f)
        assert len(self.qa_data) == subset_size
        # self.qa_data = [
        #     x for x in self.qa_data if x["episode_history"].startswith("hm3d-v0/")
        # ]
        for x in self.qa_data:
            assert x["episode_history"].startswith("hm3d-v0/"), x["episode_history"]

        # * load spawn data
        self.frame_dir = osp.join(data_dir, "frames")
        for qa_datum in self.qa_data:
            # * get the first frame
            pkl_path = osp.join(self.frame_dir, qa_datum["episode_history"], "00000.pkl")
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
                agent_state = pkl_data["agent_state"]
                sensor_states = agent_state.sensor_states

                agent_pos = agent_state.position
                agent_rot = agent_state.rotation
                rgb_pos = sensor_states["rgb"].position
                rgb_rot = sensor_states["rgb"].rotation
                dep_pos = sensor_states["depth"].position
                dep_rot = sensor_states["depth"].rotation
                sem_pos = sensor_states["semantic"].position
                sem_rot = sensor_states["semantic"].rotation

                for i in range(3):
                    assert (
                        agent_pos[i] + (i == 1)  # camera height
                        == rgb_pos[i]
                        == dep_pos[i]
                        == sem_pos[i]
                    )
                assert agent_rot == rgb_rot == dep_rot == sem_rot

                scene_id = osp.join(*(pkl_data["scene_id"].split("/")[-3:]))
                assert scene_id.startswith("val/")
                scene_id = osp.join(self.hm3d_dir, scene_id)

                datum = dict(
                    qa_datum,
                    scene_id=scene_id,
                    hfov=pkl_data["hfov"],
                    start_position=agent_pos.tolist(),
                    start_rotation=agent_rot.components.tolist(),
                )
                self.data.append(datum)

        # * finalize
        # self.data = self.filter_for_demo(self.data, self.get_demo_list())
        return self.data

    def __getitem__(self, idx):
        if self.reverse_order:
            idx = len(self.data) - 1 - idx
        return self.data[idx]
