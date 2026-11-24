import torch

from .models.stream3r import STream3R


class StreamSession:
    """
    A causal streaming inference session with KV cache management for STream3R.

    Modes:
      - "causal": unbounded KV cache (all frames retained).
      - "window": keep first 1 frame (anchor) + last `window_size` frames.
    """

    def __init__(self, model: STream3R, mode: str, window_size: int = 5):
        self.model = model
        self.mode = mode
        self.window_size = window_size
        self.aggregator_kv_cache_depth = model.aggregator.depth
        self.camera_head_kv_cache_depth = model.camera_head.trunk_depth
        self.camera_head_iterations = 4

        if self.mode not in ["causal", "window"]:
            raise ValueError(f"Unsupported attention mode: {self.mode}")

        self.clear()

    def _clear_predictions(self):
        self.predictions = {}

    def _update_predictions(self, predictions):
        for k in ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]:
            if k in predictions:
                self.predictions[k] = torch.cat(
                    [self.predictions.get(k, torch.empty(0, device=predictions[k].device)), predictions[k]], dim=1
                )

    def _clear_cache(self):
        self.aggregator_kv_cache_list = [[None, None] for _ in range(self.aggregator_kv_cache_depth)]
        self.camera_head_kv_cache_list = [
            [[None, None] for _ in range(self.camera_head_kv_cache_depth)] for _ in range(self.camera_head_iterations)
        ]

    def _update_cache(self, aggregator_kv_cache_list, camera_head_kv_cache_list):
        if self.mode == "causal":
            self.aggregator_kv_cache_list = aggregator_kv_cache_list
            self.camera_head_kv_cache_list = camera_head_kv_cache_list
        elif self.mode == "window":
            ws = self.window_size
            for k in range(2):
                for i in range(self.aggregator_kv_cache_depth):
                    h, w = self.predictions["depth"].shape[2], self.predictions["depth"].shape[3]
                    P = (
                        h * w // self.model.aggregator.patch_size // self.model.aggregator.patch_size
                        + self.model.aggregator.patch_start_idx
                    )
                    anchor_token = aggregator_kv_cache_list[i][k][:, :, :P]
                    window_tokens = aggregator_kv_cache_list[i][k][
                        :, :, max(P, aggregator_kv_cache_list[i][k].size(2) - ws * P) :
                    ]
                    self.aggregator_kv_cache_list[i][k] = torch.cat([anchor_token, window_tokens], dim=2)
                for i in range(self.camera_head_iterations):
                    for j in range(self.camera_head_kv_cache_depth):
                        anchor_token = camera_head_kv_cache_list[i][j][k][:, :, :1]
                        window_tokens = camera_head_kv_cache_list[i][j][k][
                            :, :, max(1, camera_head_kv_cache_list[i][j][k].size(2) - ws) :
                        ]
                        self.camera_head_kv_cache_list[i][j][k] = torch.cat([anchor_token, window_tokens], dim=2)

    def _get_cache(self):
        return self.aggregator_kv_cache_list, self.camera_head_kv_cache_list

    def get_all_predictions(self):
        return self.predictions

    def get_last_prediction(self):
        last_predictions = {}
        for k in ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]:
            if k in self.predictions:
                last_predictions[k] = self.predictions[k][:, -1:]
        return last_predictions

    def clear(self):
        self._clear_predictions()
        self._clear_cache()

    def forward_stream(self, images):
        aggregator_kv_cache_list, camera_head_kv_cache_list = self._get_cache()

        outputs = self.model(
            images=images,
            mode=self.mode,
            aggregator_kv_cache_list=aggregator_kv_cache_list,
            camera_head_kv_cache_list=camera_head_kv_cache_list,
        )

        self._update_predictions(outputs)
        self._update_cache(outputs["aggregator_kv_cache_list"], outputs["camera_head_kv_cache_list"])

        return self.get_all_predictions()
