import json

import numpy as np
import torch

from .scene_inputs import SceneInputs

decord_available = False
try:
    import decord
    decord.bridge.set_bridge('torch')
    decord_available = True
except ImportError:
    pass


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, **kwargs):
        self.num_frames = kwargs['min_num_frames']
        self.sample_size = kwargs['video_size']
        self.traj_txt_path = kwargs.get('traj_txt_path', None)
        self.relative_to_source = kwargs.get('relative_to_source', False)
        self.rotation_only = kwargs.get('rotation_only', False)
        self.adaptive_frame = kwargs.get('adaptive_frame', True)
        self.freeze_repeat = kwargs.get('freeze_repeat', 0)
        self.freeze_frame = kwargs.get('freeze_frame', None)

        if not isinstance(json_path, str):
            json_path = json_path[0]
        assert isinstance(json_path, str), f"json_path must be a string, got {type(json_path)}"

        self.metadata_list = json.load(open(json_path, 'r'))
        self.dataset = {entry['video_path']: entry for entry in self.metadata_list}
        for entry in self.metadata_list:
            entry['dataset_type'] = 'test'

        self.scene_inputs = SceneInputs(
            self.sample_size,
            self.num_frames,
            traj_txt_path=self.traj_txt_path,
            relative_to_source=self.relative_to_source,
            rotation_only=self.rotation_only,
            adaptive_frame=self.adaptive_frame,
            freeze_repeat=self.freeze_repeat,
            freeze_frame=self.freeze_frame,
        )
        print(f"Loaded {len(self.dataset)} videos.")

    def __getitem__(self, index):
        source_data_key = list(self.dataset.keys())[index]
        try:
            data = self.scene_inputs.get_data(self.dataset[source_data_key])
            data = self._temporal_sampling(data)
        except Exception as exc:
            raise RuntimeError(f"Failed to load inference input {source_data_key}") from exc
        data['index'] = index
        return data

    def _temporal_sampling(self, data):
        current_frames = data['source_video'].shape[0]
        target_frames = self.num_frames
        if target_frames < current_frames:
            indices = torch.linspace(0, current_frames - 1, target_frames).long()
            for key in list(data.keys()):
                if key in data and isinstance(data[key], (torch.Tensor, np.ndarray)):
                    data[key] = data[key][indices]
        return data

    def __len__(self):
        return len(self.dataset)
