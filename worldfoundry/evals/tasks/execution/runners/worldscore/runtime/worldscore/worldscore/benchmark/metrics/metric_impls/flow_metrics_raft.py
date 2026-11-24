import argparse
import itertools

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.perception_core.optical_flow.raft import (
    RAFT,
    InputPadder,
    checkpoint_path,
)
from worldfoundry.evals.tasks.execution.runners.worldscore.runtime.worldscore.worldscore.benchmark.metrics.base_metrics import (
    BaseMetric,
)


class OpticalFlowMetric(BaseMetric):
    """

    Using the median of estimated optical-flow to measure the motion magnitude.

    Optical-flow estimation -- RAFT

    RANGE: [0, ~] higher the better
    """

    def __init__(self) -> None:
        super().__init__()

        args = {
            "model": str(checkpoint_path()),
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        }
        args = argparse.Namespace(**args)

        # load model
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        model = model.module
        model.to(self._device)
        model.eval()
        self._model = model
        self._args = args

    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self._device)

    def _compute_flow(self, image1, image2):
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        with torch.amp.autocast(device_type="cuda"):
            _, flow_up = self._model(image1, image2, iters=20, test_mode=True)

        flow = flow_up.cpu().numpy().squeeze().transpose(1, 2, 0)

        return flow

    def _compute_scores(
        self,
        rendered_images: list[str],
    ) -> float:
        scores = []

        with torch.no_grad():
            images = rendered_images

            for _i, (imfile1, imfile2) in tqdm(
                enumerate(itertools.pairwise(images)), total=len(images) - 1, desc="Computing flow..."
            ):
                image1 = self.load_image(imfile1)
                image2 = self.load_image(imfile2)

                flow = self._compute_flow(image1, image2)
                flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                median_flow = float(torch.from_numpy(flow_magnitude).median().item())
                scores.append(median_flow)

        score = sum(scores) / len(scores)
        return score
