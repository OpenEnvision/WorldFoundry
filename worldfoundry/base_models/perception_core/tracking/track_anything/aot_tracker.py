# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> perception_core -> tracking -> track_anything -> aot_tracker.py functionality."""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .aot import config as engine_config
from .aot.networks.engines import build_engine
from .aot.networks.engines.aot_engine import AOTEngine, AOTInferEngine
from .aot.networks.engines.deaot_engine import DeAOTEngine, DeAOTInferEngine
from .aot.networks.models import build_vos_model
from .aot.transforms import video_transforms as tr
from .aot.utils.checkpoint import load_network


class AOTTracker(object):
    """Aot tracker implementation."""
    def __init__(self, cfg, gpu_id=0):
        """Init.

        Args:
            cfg: The cfg.
            gpu_id: The gpu id.
        """
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.engine = build_engine(
            cfg.MODEL_ENGINE,
            phase="eval",
            aot_model=self.model,
            gpu_id=gpu_id,
            short_term_mem_skip=1,
            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
            max_len_long_term=cfg.MAX_LEN_LONG_TERM,
        )

        self.transform = transforms.Compose(
            [
                tr.MultiRestrictSize(
                    cfg.TEST_MAX_SHORT_EDGE,
                    cfg.TEST_MAX_LONG_EDGE,
                    cfg.TEST_FLIP,
                    cfg.TEST_MULTISCALE,
                    cfg.MODEL_ALIGN_CORNERS,
                ),
                tr.MultiToTensor(),
            ]
        )

        self.model.eval()

    @torch.no_grad()
    def add_reference_frame(self, frame, mask, obj_nums, frame_step, incremental=False):
        """Add reference frame.

        Args:
            frame: The frame.
            mask: The mask.
            obj_nums: The obj nums.
            frame_step: The frame step.
            incremental: The incremental.
        """
        # mask = cv2.resize(mask, frame.shape[:2][::-1], interpolation = cv2.INTER_NEAREST)

        sample = {
            "current_img": frame,
            "current_label": mask,
        }

        sample = self.transform(sample)
        frame = sample[0]["current_img"].unsqueeze(0).float().cuda(self.gpu_id)
        mask = sample[0]["current_label"].unsqueeze(0).float().cuda(self.gpu_id)
        _mask = F.interpolate(mask, size=frame.shape[-2:], mode="nearest")

        if incremental:
            self.engine.add_reference_frame_incremental(frame, _mask, obj_nums=obj_nums, frame_step=frame_step)
        else:
            self.engine.add_reference_frame(frame, _mask, obj_nums=obj_nums, frame_step=frame_step)

    @torch.no_grad()
    def track(self, image):
        """Track.

        Args:
            image: The image.
        """
        output_height, output_width = image.shape[0], image.shape[1]
        sample = {"current_img": image}
        sample = self.transform(sample)
        image = sample[0]["current_img"].unsqueeze(0).float().cuda(self.gpu_id)
        self.engine.match_propogate_one_frame(image)
        pred_logit = self.engine.decode_current_logits((output_height, output_width))

        # pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_logit, dim=1, keepdim=True).float()

        return pred_label

    @torch.no_grad()
    def update_memory(self, pred_label):
        """Update memory.

        Args:
            pred_label: The pred label.
        """
        self.engine.update_memory(pred_label)

    @torch.no_grad()
    def restart(self):
        """Restart."""
        self.engine.restart_engine()

    @torch.no_grad()
    def build_tracker_engine(self, name, **kwargs):
        """Build tracker engine.

        Args:
            name: The name.
        """
        if name == "aotengine":
            return AOTTrackerInferEngine(**kwargs)
        elif name == "deaotengine":
            return DeAOTTrackerInferEngine(**kwargs)
        else:
            raise NotImplementedError


class AOTTrackerInferEngine(AOTInferEngine):
    """Aot tracker infer engine implementation."""
    def __init__(
        self,
        aot_model,
        gpu_id=0,
        long_term_mem_gap=9999,
        short_term_mem_skip=1,
        max_aot_obj_num=None,
    ):
        """Init.

        Args:
            aot_model: The aot model.
            gpu_id: The gpu id.
            long_term_mem_gap: The long term mem gap.
            short_term_mem_skip: The short term mem skip.
            max_aot_obj_num: The max aot obj num.
        """
        super().__init__(aot_model, gpu_id, long_term_mem_gap, short_term_mem_skip, max_aot_obj_num)

    def add_reference_frame_incremental(self, img, mask, obj_nums, frame_step=-1):
        """Add reference frame incremental.

        Args:
            img: The img.
            mask: The mask.
            obj_nums: The obj nums.
            frame_step: The frame step.
        """
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while aot_num > len(self.aot_engines):
            new_engine = AOTEngine(self.AOT, self.gpu_id, self.long_term_mem_gap, self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(self.aot_engines, separated_masks, separated_obj_nums):
            if aot_engine.obj_nums is None or aot_engine.obj_nums[0] < separated_obj_num:
                aot_engine.add_reference_frame(
                    img,
                    separated_mask,
                    obj_nums=[separated_obj_num],
                    frame_step=frame_step,
                    img_embs=img_embs,
                )
            else:
                aot_engine.update_short_term_memory(separated_mask)

            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()


class DeAOTTrackerInferEngine(DeAOTInferEngine):
    """De aot tracker infer engine implementation."""
    def __init__(
        self,
        aot_model,
        gpu_id=0,
        long_term_mem_gap=9999,
        short_term_mem_skip=1,
        max_aot_obj_num=None,
    ):
        """Init.

        Args:
            aot_model: The aot model.
            gpu_id: The gpu id.
            long_term_mem_gap: The long term mem gap.
            short_term_mem_skip: The short term mem skip.
            max_aot_obj_num: The max aot obj num.
        """
        super().__init__(aot_model, gpu_id, long_term_mem_gap, short_term_mem_skip, max_aot_obj_num)

    def add_reference_frame_incremental(self, img, mask, obj_nums, frame_step=-1):
        """Add reference frame incremental.

        Args:
            img: The img.
            mask: The mask.
            obj_nums: The obj nums.
            frame_step: The frame step.
        """
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while aot_num > len(self.aot_engines):
            new_engine = DeAOTEngine(self.AOT, self.gpu_id, self.long_term_mem_gap, self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(self.aot_engines, separated_masks, separated_obj_nums):
            if aot_engine.obj_nums is None or aot_engine.obj_nums[0] < separated_obj_num:
                aot_engine.add_reference_frame(
                    img,
                    separated_mask,
                    obj_nums=[separated_obj_num],
                    frame_step=frame_step,
                    img_embs=img_embs,
                )
            else:
                aot_engine.update_short_term_memory(separated_mask)

            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()


def get_aot(args):
    """Get aot.

    Args:
        args: The args.
    """
    # build vos engine
    cfg = engine_config.EngineConfig(args["phase"])
    cfg.TEST_CKPT_PATH = args["model_path"]
    cfg.TEST_LONG_TERM_MEM_GAP = args["long_term_mem_gap"]
    cfg.MAX_LEN_LONG_TERM = args["max_len_long_term"]
    tracker = AOTTracker(cfg, args["gpu_id"])
    return tracker
