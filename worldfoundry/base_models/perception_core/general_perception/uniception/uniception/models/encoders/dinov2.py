"""
Encoder Class for DINOv2
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.core.model_loading.checkpoints import load_tensor_state_dict

from uniception.models.encoders.base import UniCeptionViTEncoderBase, ViTEncoderInput, ViTEncoderOutput
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner


class DINOv2Encoder(UniCeptionViTEncoderBase):
    "UniCeption DINOv2 Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "dinov2",
        patch_size: int = 14,
        size: str = "large",
        with_registers: bool = False,
        norm_returned_features: bool = True,
        pretrained_checkpoint_path: str = None,
        torch_hub_force_reload: bool = False,
        torch_hub_pretrained: bool = True,
        gradient_checkpointing: bool = False,
        keep_first_n_layers: Optional[int] = None,
        use_pytorch_sdpa=True,
        disable_torch_compile_for_pe=False,
        *args,
        **kwargs,
    ):
        """
        DINOv2 Encoder for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Image normalization type. Default: "dinov2"
            patch_size (int): Patch size for the encoder. Default: 14
            size (str): Size variant of the DINOv2 model. Options: ["small", "base", "large", "giant"]. Default: "large"
            with_registers (bool): Whether to use the DINOv2 model with registers. Default: False
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of DINOv2. Default: None
            torch_hub_force_reload (bool): Whether to force reload the model from torch hub. Default: False
            torch_hub_pretrained (bool): Whether to use the pretrained weights from torch hub. Default: True
            gradient_checkpointing (bool): Whether to use gradient checkpointing to save GPU memory during backward call. Default: False
            keep_first_n_layers (Optional[int]): If specified, only the first n layers of the model will be kept. Default: None
            use_pytorch_sdpa (bool): Whether to use PyTorch native SDPA for attention layers. Default: True
            disable_torch_compile_for_pe (bool): Whether to disable torch compile for PE interpolation. Default: False
        """
        # Init the base class
        name = name if not with_registers else f"{name}_reg"
        super().__init__(
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            gradient_checkpointing=gradient_checkpointing,
            *args,
            **kwargs,
        )

        # Init the DINOv2 Encoder specific attributes
        self.version = size
        self.with_registers = with_registers
        self.norm_returned_features = norm_returned_features
        self.enc_embed_dim = {"small": 384, "base": 768, "large": 1024, "giant": 1536}[self.version]

        # Define DINOv2 model factory
        DINO_MODELS = {
            # No registers
            False: {
                "small": "dinov2_vits14",
                "base": "dinov2_vitb14",
                "large": "dinov2_vitl14",
                "giant": "dinov2_vitg14",
            },
            # With registers
            True: {
                "small": "dinov2_vits14_reg",
                "base": "dinov2_vitb14_reg",
                "large": "dinov2_vitl14_reg",
                "giant": "dinov2_vitg14_reg",
            },
        }

        # Reuse WorldFoundry's central DINOv2 implementation. Only model weights
        # may be fetched into the normal torch cache; no source repository is
        # downloaded or imported at runtime.
        model_name = DINO_MODELS[self.with_registers][self.version]
        print(f"Loading pretrained {model_name} from WorldFoundry's in-tree DINOv2 base model")
        try:
            from worldfoundry.base_models.perception_core.general_perception.dinov2.hub import (
                backbones as in_tree_dinov2,
            )
        except ImportError as exc:
            raise ImportError(
                "UniCeption requires WorldFoundry's in-tree DINOv2 base model on PYTHONPATH"
            ) from exc
        model_factory = getattr(in_tree_dinov2, model_name)
        self.model = model_factory(
            pretrained=torch_hub_pretrained if pretrained_checkpoint_path is None else False,
        )

        del (
            self.model.mask_token
        )  # This parameter is unused in producing patch features, and will lead to unused parameters

        # Disable position interpolation at PE interpolation to enable torch compile
        # when training with multiple image shapes.
        if disable_torch_compile_for_pe:
            self.model.interpolate_pos_encoding = torch.compiler.disable(self.model.interpolate_pos_encoding)

        if not norm_returned_features:
            self.model.norm = nn.Identity()  # Drop final normalization if not desired

        # Keep only the first n layers of the model if keep_first_n_layers is specified
        if keep_first_n_layers is not None:
            self.model.blocks = nn.ModuleList(self.model.blocks[:keep_first_n_layers])

        # Use Native Torch SDPA for attention layers if specified (instead of DINOv2's XFormers)
        if use_pytorch_sdpa:
            self.enable_pytorch_native_sdpa()

        # Wrap the transformer blocks with support for gradient checkpointing if required
        if self.gradient_checkpointing:
            for i in range(len(self.model.blocks)):
                self.model.blocks[i] = self.wrap_module_with_gradient_checkpointing(self.model.blocks[i])

        # Load the custom pretrained checkpoint if provided
        if pretrained_checkpoint_path:
            print(f"Loading custom pretrained DINOv2 checkpoint from {pretrained_checkpoint_path}")
            state_dict = load_tensor_state_dict(
                pretrained_checkpoint_path,
                wrapper_keys=("model", "state_dict", "model_state_dict", "module"),
            )
            print(self.load_state_dict(state_dict))

    def enable_pytorch_native_sdpa(self):
        "Enable PyTorch native SDPA for attention layers"
        for i in range(len(self.model.blocks)):
            self.model.blocks[i].attn = self.wrap_dinov2_attention_with_sdpa(self.model.blocks[i].attn)

    def wrap_dinov2_attention_with_sdpa(self, module: nn.Module):
        "Wrap DINOv2 attention module with PyTorch native SDPA"
        assert torch.__version__ >= "2.0", "SDPA requires PyTorch 2.0 or later"

        class _AttentionWrapper(module.__class__):
            "SDPA Attention Wrapper Class"

            def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
                B, N, C = x.shape
                qkv = (
                    self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                )  # (3, B, H, N, C // H)

                q, k, v = torch.unbind(qkv, 0)  # (B, H, N, C // H)

                x = F.scaled_dot_product_attention(q, k, v, attn_bias)
                x = x.permute(0, 2, 1, 3).reshape(B, N, C)

                x = self.proj(x)
                x = self.proj_drop(x)
                return x

        module.__class__ = _AttentionWrapper
        return module

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        DINOv2 Encoder Forward Pass

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            ViTEncoderOutput: Output data from the encoder.
        """
        # Check image normalization type
        self._check_data_normalization_type(encoder_input.data_norm_type)

        # Check the dtype and shape of the input image
        assert isinstance(encoder_input.image, torch.Tensor), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        # Extract the features from the DINOv2 model
        result_dict = self.model.forward_features(encoder_input.image)

        # Patch tokens
        features = result_dict["x_norm_patchtokens"]

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        # Additional registers (including cls token) if present
        additional_registers = []

        # Add the cls token
        cls_token = result_dict["x_norm_clstoken"].unsqueeze(1)  # (B x 1 x Embed_dim)
        additional_registers.append(cls_token)

        # Add the registers
        registers = result_dict["x_norm_regtokens"]
        if registers is not None:
            additional_registers.append(registers)

        all_registers = torch.cat(additional_registers, dim=1) if len(additional_registers) > 0 else None
        if all_registers is not None:
            all_registers = all_registers.permute(0, 2, 1).contiguous()  # (B x Embed_dim x Num_registers)

        return ViTEncoderOutput(features=features, registers=all_registers)


class DINOv2IntermediateFeatureReturner(DINOv2Encoder, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption DINOv2 Encoder"

    def __init__(
        self,
        name: str,
        data_norm_type: str = "dinov2",
        patch_size: int = 14,
        size: str = "large",
        with_registers: bool = False,
        pretrained_checkpoint_path: str = None,
        torch_hub_force_reload: bool = False,
        gradient_checkpointing: bool = False,
        keep_first_n_layers: Optional[int] = None,
        use_pytorch_sdpa=True,
        disable_torch_compile_for_pe=False,
        indices: Optional[Union[int, List[int]]] = 1,
        norm_intermediate: bool = True,
        *args,
        **kwargs,
    ):
        """
        DINOv2 Encoder for extracting spatial features from images.

        Args:
            name (str): Name of the encoder.
            data_norm_type (str): Image normalization type. Default: "dinov2"
            patch_size (int): Patch size for the encoder. Default: 14
            size (str): Size variant of the DINOv2 model. Options: ["small", "base", "large", "giant"]. Default: "large"
            with_registers (bool): Whether to use the DINOv2 model with registers. Default: False
            pretrained_checkpoint_path (str): Path to the pretrained checkpoint if using custom trained version of DINOv2. Default: None
            torch_hub_force_reload (bool): Whether to force reload the model from torch hub. Default: False
            gradient_checkpointing (bool): Whether to use gradient checkpointing to save GPU memory during backward call. Default: False
            keep_first_n_layers (Optional[int]): If specified, only the first n layers of the model will be kept. Default: None
            use_pytorch_sdpa (bool): Whether to use PyTorch native SDPA for attention layers. Default: True
            disable_torch_compile_for_pe (bool): Whether to disable torch compile for PE interpolation. Default: False
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to 1. Options:
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
        """
        # Init the base classes
        DINOv2Encoder.__init__(
            self,
            name=name,
            data_norm_type=data_norm_type,
            patch_size=patch_size,
            size=size,
            with_registers=with_registers,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            torch_hub_force_reload=torch_hub_force_reload,
            gradient_checkpointing=gradient_checkpointing,
            keep_first_n_layers=keep_first_n_layers,
            use_pytorch_sdpa=use_pytorch_sdpa,
            disable_torch_compile_for_pe=disable_torch_compile_for_pe,
            *args,
            **kwargs,
        )
        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
        )

    def forward(self, encoder_input: ViTEncoderInput) -> List[ViTEncoderOutput]:
        """
        DINOv2 Encoder Forward Pass with Intermediate Feature Return

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            List[ViTEncoderOutput]: Output data from the encoder. Returns a list of intermediate features.
        """
        # Check image normalization type
        self._check_data_normalization_type(encoder_input.data_norm_type)

        # Check the dtype and shape of the input image
        assert isinstance(encoder_input.image, torch.Tensor), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        if self.indices is None:
            self.indices = range(len(self.model.blocks))

        # Extract the intermediate features from the DINOv2 model
        intermediate_features = self.model.get_intermediate_layers(
            encoder_input.image, n=self.indices, reshape=True, norm=self.norm_intermediate, return_class_token=True
        )

        # Convert the intermediate features to a list of ViTEncoderOutput
        intermediate_features = [
            ViTEncoderOutput(features=features, registers=cls_tokens.unsqueeze(-1))
            for features, cls_tokens in intermediate_features
        ]

        return intermediate_features
