from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.device import resolve_inference_device
from worldfoundry.core.io import load_video_frames
from worldfoundry.core.utils import freeze_params as freeze_backbone

from .registry import MODEL


def _opencv():
    import cv2

    return cv2


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    from torchvision.transforms.functional import to_tensor

    return to_tensor(image)


def _inference_device() -> torch.device:
    requested = os.environ.get("LARY_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(resolve_inference_device(requested, allow_cpu_fallback=True))


def _model_path(filename):
    root = os.environ.get("MODEL_DIR")
    if not root:
        raise ValueError("MODEL_DIR must point to the LARYBench checkpoint directory")
    return os.path.join(root, filename)


def _video_sample(data, index):
    row = data.iloc[index]
    frame_indices = [int(value) for value in str(row["sample_indices"]).split(",")]
    relative_indices = torch.tensor([value - frame_indices[0] for value in frame_indices], dtype=torch.long)
    frames = load_video_frames(row["video_path"])
    if len(frames) == 0:
        return None
    last = len(frames) - 1
    selected = [frames[max(0, min(value, last))] for value in frame_indices]
    return data.index[index], selected, relative_indices


def _cv_image(path, image_size):
    cv2 = _opencv()
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return image / 255.0


class LaryBaseModel(ABC):
    name: str

    @abstractmethod
    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        raise NotImplementedError

    @abstractmethod
    def prepare_model_for_extraction(self):
        raise NotImplementedError

    @abstractmethod
    def process_image(self, src_img_path, tgt_img_path, image_size):
        raise NotImplementedError

    @abstractmethod
    def process_video(self, data: Any, index, image_size):
        raise NotImplementedError


class LaqBaseModel(LaryBaseModel):
    def prepare_model_for_extraction(self):
        freeze_backbone(self.model)
        self.model.to(_inference_device()).eval()

    def _get_latent_action(self, batch_input):
        tokens, indices = self.model(batch_input, return_only_codebook_ids=True)
        return tokens.cpu().numpy(), indices.cpu().numpy()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        if batch_data.ndim == 6:
            batch_size, pair_count, channels, frames, height, width = batch_data.shape
            flat = batch_data.view(batch_size * pair_count, channels, frames, height, width).to(
                _inference_device()
            )
            tokens, indices = self._get_latent_action(flat)
            tokens = tokens.reshape(batch_size, pair_count, tokens.shape[-2], tokens.shape[-1])
            indices = indices.reshape(batch_size, pair_count, -1)
            return tokens, indices
        return self._get_latent_action(batch_data.to(_inference_device()))

    def process_image(self, src_img_path, tgt_img_path, image_size):
        pair = np.stack([_cv_image(src_img_path, image_size), _cv_image(tgt_img_path, image_size)])
        return torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)

    def process_video(self, data, index, image_size):
        sample = _video_sample(data, index)
        if sample is None:
            return None
        global_index, frames, relative_indices = sample
        cv2 = _opencv()
        frames = [
            cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_CUBIC) / 255.0 for frame in frames
        ]
        pairs = np.stack([np.stack([first, second]) for first, second in zip(frames, frames[1:])])
        tensor = torch.tensor(pairs, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        return global_index, tensor, relative_indices


def _initialize_laq(instance, model_class, name, checkpoint, args, kwargs):
    instance.name = kwargs.pop("name", name)
    instance.model = model_class(*args, **kwargs)
    instance.model.load(_model_path(checkpoint))
    instance.prepare_model_for_extraction()


@MODEL.register_module()
class LAPALaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.laq import (
            LatentActionQuantization,
        )

        _initialize_laq(self, LatentActionQuantization, "lapa", "laq_openx.pt", args, kwargs)


@MODEL.register_module()
class Magvit2LaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.laq import (
            LatentActionQuantizationMagvit2,
        )

        _initialize_laq(
            self,
            LatentActionQuantizationMagvit2,
            "magvit2",
            "magvit2.pt",
            args,
            kwargs,
        )


@MODEL.register_module()
class Dinov2LaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.laq import (
            LatentActionQuantizationDinov2Feature,
        )

        _initialize_laq(
            self,
            LatentActionQuantizationDinov2Feature,
            "dinov2",
            "laq_dinov2.pt",
            args,
            kwargs,
        )


@MODEL.register_module()
class Dinov3LaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.laq import (
            LatentActionQuantizationDinov3Feature,
        )

        self.name = kwargs.pop("name", "dinov3")
        self.model = LatentActionQuantizationDinov3Feature(*args, **kwargs)
        checkpoint = torch.load(_model_path("laq_dinov3.pt"), map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.prepare_model_for_extraction()


@MODEL.register_module()
class Siglip2LaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.laq import (
            LatentActionQuantizationSiglipv2Feature,
        )

        _initialize_laq(
            self,
            LatentActionQuantizationSiglipv2Feature,
            "siglip2",
            "siglip2.pt",
            args,
            kwargs,
        )


@MODEL.register_module()
class UnivlaLaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.univla import (
            ControllableDINOLatentActionModel,
        )

        self.name = kwargs.pop("name", "univla")
        checkpoint_path = os.environ.get("UNIVLA_CKPT_PATH") or _model_path(
            "univla-latent-action-model/lam-stage-2.ckpt"
        )
        self.model = ControllableDINOLatentActionModel(*args, **kwargs)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict(
            {key.removeprefix("lam."): value for key, value in checkpoint.items()},
            strict=False,
        )
        self.prepare_model_for_extraction()

    def _get_latent_action(self, batch_input):
        outputs = self.model.vq_encode(batch_input.permute(0, 2, 1, 3, 4))
        return (
            outputs["z_q"].squeeze(1).cpu().numpy(),
            outputs["indices"].cpu().numpy(),
        )


@MODEL.register_module()
class VillaXLaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.villa_x import (
            IgorModel,
        )

        self.name = kwargs.pop("name", "villa-x")
        self.model = IgorModel.from_pretrained(os.environ.get("VILLA_X_CKPT_PATH"), strict=False)
        self.prepare_model_for_extraction()

    def prepare_model_for_extraction(self):
        freeze_backbone(self.model)
        self.model.to(_inference_device()).eval()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        output = self.model.idm(batch_data.to(_inference_device()), return_dict=True)
        tokens = output["vq_tokens"].cpu().numpy()
        if config.mode == "image":
            tokens = tokens.reshape(tokens.shape[0], -1, self.model.config.action_latent_dim)
        else:
            tokens = tokens.reshape(tokens.shape[0], 8, -1, self.model.config.action_latent_dim)
        indices = output["indices"].cpu().numpy().reshape(tokens.shape[0], tokens.shape[1], -1)
        return tokens, indices

    @staticmethod
    def _load_image(path, image_size):
        image = Image.open(path).convert("RGB")
        return np.asarray(image.resize((image_size, image_size), Image.LANCZOS))

    def process_image(self, src_img_path, tgt_img_path, image_size):
        return torch.tensor(
            np.stack(
                [
                    self._load_image(src_img_path, image_size),
                    self._load_image(tgt_img_path, image_size),
                ]
            ),
            dtype=torch.uint8,
        )

    def process_video(self, data, index, image_size):
        sample = _video_sample(data, index)
        if sample is None:
            return None
        global_index, frames, relative_indices = sample
        cv2 = _opencv()
        frames = [cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_CUBIC) for frame in frames]
        return global_index, torch.from_numpy(np.stack(frames)), relative_indices


@MODEL.register_module()
class Flux2LaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.diffusion_model.image.flux2 import load_autoencoder

        self.name = kwargs.pop("name", "flux2")
        self.model = load_autoencoder(device=str(_inference_device()))
        self.prepare_model_for_extraction()

    def prepare_model_for_extraction(self):
        freeze_backbone(self.model)
        self.model.to(_inference_device()).eval()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        from worldfoundry.base_models.diffusion_model.image.flux2 import (
            encode_video_batch_refs,
        )

        latents = encode_video_batch_refs(self.model, batch_data)
        tokens = latents.permute(0, 1, 3, 4, 2).flatten(2, 3).cpu().numpy()
        return tokens, [np.array([]) for _ in tokens]

    @staticmethod
    def _load_image(path, image_size):
        image = Image.open(path).convert("RGB")
        return (
            np.asarray(
                image.resize((image_size, image_size), Image.LANCZOS),
                dtype=np.float32,
            )
            / 255.0
        )

    def process_image(self, src_img_path, tgt_img_path, image_size):
        frames = np.stack(
            [
                self._load_image(src_img_path, image_size),
                self._load_image(tgt_img_path, image_size),
            ]
        )
        return torch.tensor(frames, dtype=torch.float32) * 2 - 1

    def process_video(self, data, index, image_size):
        sample = _video_sample(data, index)
        if sample is None:
            return None
        global_index, frames, relative_indices = sample
        cv2 = _opencv()
        frames = np.stack(
            [cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_CUBIC) / 255.0 for frame in frames]
        )
        return (
            global_index,
            torch.tensor(frames, dtype=torch.float32) * 2 - 1,
            relative_indices,
        )


@MODEL.register_module()
class Wan2_2LaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_2 import (
            Wan2_2_VAE,
        )

        self.name = kwargs.pop("name", "wan2-2")
        self.model = Wan2_2_VAE(
            vae_pth=os.environ.get("WAN22_VAE_PATH"),
            device=str(_inference_device()),
        )
        self.prepare_model_for_extraction()

    def prepare_model_for_extraction(self):
        freeze_backbone(self.model.model)
        self.model.model.to(_inference_device()).eval()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        latents = self.model.encode([value.to(_inference_device()) for value in batch_data])
        tokens = []
        for latent in latents:
            value = np.transpose(latent.cpu().numpy(), (1, 2, 3, 0))
            tokens.append(value.reshape(value.shape[0], -1, value.shape[-1]))
        return tokens, [np.array([]) for _ in tokens]

    @staticmethod
    def _load_image(path, image_size):
        image = Image.open(path).convert("RGB")
        image = image.resize((image_size, image_size), Image.LANCZOS)
        return _image_to_tensor(image).sub_(0.5).div_(0.5)

    def process_image(self, src_img_path, tgt_img_path, image_size):
        return torch.stack(
            [
                self._load_image(src_img_path, image_size),
                self._load_image(tgt_img_path, image_size),
            ],
            dim=1,
        )

    def process_video(self, data, index, image_size):
        sample = _video_sample(data, index)
        if sample is None:
            return None
        global_index, frames, relative_indices = sample
        tensors = []
        for frame in frames:
            image = Image.fromarray(frame).resize((image_size, image_size), Image.LANCZOS)
            tensors.append(_image_to_tensor(image).sub_(0.5).div_(0.5))
        return global_index, torch.stack(tensors, dim=1), relative_indices


class VjepaBaseModel(LaryBaseModel):
    def prepare_model_for_extraction(self):
        freeze_backbone(self.model)
        self.model.to(_inference_device()).eval()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        if config.mode == "video":
            transformed = [
                torch.stack(self.transform(frames), dim=0).to(_inference_device(), non_blocking=True)
                for frames in batch_data
            ]
            clips = [[torch.stack(transformed).squeeze(1)]]
            clip_indices = batch_rel_indices.to(_inference_device())
        else:
            clips = [
                [
                    torch.stack([self.transform(frames)[0] for frames in batch_data]).to(
                        _inference_device(), non_blocking=True
                    )
                ]
            ]
            clip_indices = torch.tensor([0, config.stride], device=_inference_device()).repeat(
                len(batch_data), 1
            )
        tokens = self.model(clips, clip_indices)[0].cpu().numpy()
        return tokens, [np.array([]) for _ in tokens]

    @staticmethod
    def process_image(src_img_path, tgt_img_path, image_size):
        return [
            np.asarray(Image.open(src_img_path).convert("RGB")),
            np.asarray(Image.open(tgt_img_path).convert("RGB")),
        ]

    def process_video(self, data, index, image_size):
        return _video_sample(data, index)


def _initialize_vjepa(instance, name, version, resolution, env_name):
    from worldfoundry.base_models.perception_core.action_recognition.latent_action.vjepa2 import (
        EvalVideoTransform,
        load_vjepa2,
    )

    instance.name = name
    instance.model = load_vjepa2(
        os.environ.get(env_name),
        version=version,
        resolution=resolution,
        frames=16,
    )
    instance.transform = EvalVideoTransform(resolution)
    instance.prepare_model_for_extraction()


@MODEL.register_module()
class Vjepa2LaryWrap(VjepaBaseModel):
    def __init__(self, *args, **kwargs):
        _initialize_vjepa(self, "vjepa2", "2", 224, "VJEPA2_CKPT_PATH")


@MODEL.register_module()
class Vjepa21LaryWrap(VjepaBaseModel):
    def __init__(self, *args, **kwargs):
        _initialize_vjepa(self, "vjepa2-1", "2.1", 384, "VJEPA21_CKPT_PATH")


class OriginBackboneLaryWrap(LaryBaseModel):
    def _initialize(self, name, tokenizer, representation):
        self.name = name
        self.model = tokenizer(device=_inference_device())
        self.representation = representation
        self.prepare_model_for_extraction()

    def prepare_model_for_extraction(self):
        freeze_backbone(self.model)
        self.model.to(_inference_device()).eval()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        batch_input = batch_data.to(_inference_device())
        batch_size, channels, frame_count, height, width = batch_input.shape
        batch_input = batch_input.reshape(batch_size * frame_count, channels, 1, height, width)
        representation = self.representation(batch_input, self.model)
        tokens = representation.reshape(batch_size, frame_count, -1, representation.shape[-1]).cpu().numpy()
        return tokens, [np.array([]) for _ in tokens]

    def process_image(self, src_img_path, tgt_img_path, image_size):
        pair = np.stack([_cv_image(src_img_path, image_size), _cv_image(tgt_img_path, image_size)])
        return torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)

    def process_video(self, data, index, image_size):
        sample = _video_sample(data, index)
        if sample is None:
            return None
        global_index, frames, relative_indices = sample
        cv2 = _opencv()
        frames = np.stack(
            [cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_CUBIC) / 255.0 for frame in frames]
        )
        tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2)
        return global_index, tensor, relative_indices


@MODEL.register_module()
class Dinov2OriginLaryWrap(OriginBackboneLaryWrap):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.backbones import (
            get_dino_reps,
            get_dino_tokenizer,
        )

        self._initialize(kwargs.pop("name", "dinov2-origin"), get_dino_tokenizer, get_dino_reps)


@MODEL.register_module()
class Dinov3OriginLaryWrap(OriginBackboneLaryWrap):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.backbones import (
            get_dinov3_reps,
            get_dinov3_tokenizer,
        )

        self._initialize(
            kwargs.pop("name", "dinov3-origin"),
            get_dinov3_tokenizer,
            get_dinov3_reps,
        )


@MODEL.register_module()
class Siglip2OriginLaryWrap(OriginBackboneLaryWrap):
    def __init__(self, *args, **kwargs):
        from worldfoundry.base_models.perception_core.action_recognition.latent_action.backbones import (
            get_siglip2_reps,
            get_siglip2_tokenizer,
        )

        self._initialize(
            kwargs.pop("name", "siglip2-origin"),
            get_siglip2_tokenizer,
            get_siglip2_reps,
        )
