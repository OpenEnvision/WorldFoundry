# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts of the canonical LTX components that serve LTX-2.5 split checkpoints."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.model_configurator import (  # noqa: E402
    VocoderConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.vocoder import VocoderWithBWE  # noqa: E402
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.model_configurator import (  # noqa: E402
    VideoEncoderConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.ltx_configurator import (  # noqa: E402
    LTXModelConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_connector import (  # noqa: E402
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.tokenizer import LTXVGemmaTokenizer  # noqa: E402
from worldfoundry.base_models.diffusion_model.models.networks.ltx.duration_head import (  # noqa: E402
    DurationHeadConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality  # noqa: E402
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel, LTXModelType  # noqa: E402
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import (  # noqa: E402
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.tiling import (  # noqa: E402
    DimensionSizeConfig,
    TileSizeConfig,
    split_by_size,
    split_temporal_causal,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import (  # noqa: E402
    SpatioTemporalScaleFactors,
)
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.model_configurator import (  # noqa: E402
    LatentUpsamplerConfigurator,
)
from worldfoundry.core.nn import DEFAULT_TRANSFORMER_OPS, PytorchAdaZeroFunction  # noqa: E402


def _small_model(
    model_type: LTXModelType = LTXModelType.AudioVideo,
    *,
    use_prompt_adaln_single: bool = True,
    ff_bias: bool = True,
    audio_ff_bias: bool = True,
    use_keyframes_abs_pos_embedding: bool = False,
) -> LTXModel:
    return LTXModel(
        model_type=model_type,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        num_layers=2,
        cross_attention_dim=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_cross_attention_dim=8,
        cross_attention_adaln=True,
        use_prompt_adaln_single=use_prompt_adaln_single,
        ff_bias=ff_bias,
        audio_ff_bias=audio_ff_bias,
        use_keyframes_abs_pos_embedding=use_keyframes_abs_pos_embedding,
    )


def _ff_biases(model: LTXModel) -> tuple[bool, bool]:
    block = model.transformer_blocks[0]
    video_ff, audio_ff = block.ff, block.audio_ff
    assert isinstance(video_ff, torch.nn.Module) and isinstance(audio_ff, torch.nn.Module)
    return (
        all(module.bias is not None for module in video_ff.modules() if isinstance(module, torch.nn.Linear)),
        all(module.bias is not None for module in audio_ff.modules() if isinstance(module, torch.nn.Linear)),
    )


def test_ltx_model_defaults_keep_ltx23_layout() -> None:
    model = _small_model()

    assert _ff_biases(model) == (True, True)
    assert model.prompt_adaln_single is not None
    assert model.audio_prompt_adaln_single is not None
    assert model.keyframes_abs_pos_embedding is None


def test_ltx_model_ltx25_flags_change_only_their_parameters() -> None:
    model = _small_model(
        use_prompt_adaln_single=False,
        ff_bias=False,
        audio_ff_bias=False,
        use_keyframes_abs_pos_embedding=True,
    )

    assert _ff_biases(model) == (False, False)
    assert model.prompt_adaln_single is None
    assert model.audio_prompt_adaln_single is None
    embedding = model.keyframes_abs_pos_embedding
    assert embedding is not None
    assert embedding.shape == (1, 16)
    assert torch.count_nonzero(embedding) == 0


def test_perturbation_types_index_the_batched_keep_masks() -> None:
    assert [int(kind) for kind in PerturbationType] == [0, 1, 2, 3]
    config = BatchedPerturbationConfig(
        [
            PerturbationConfig([Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, [1])]),
            PerturbationConfig([Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, None)]),
        ],
        num_blocks=2,
        dtype=torch.float32,
    )

    assert config.block_masks.shape == (4, 2, 2)
    assert config.block_masks[PerturbationType.SKIP_VIDEO_SELF_ATTN].tolist() == [[1.0, 1.0], [0.0, 1.0]]
    assert config.block_masks[PerturbationType.SKIP_A2V_CROSS_ATTN].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert config.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, 1)
    assert not config.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, 1)
    assert not config.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, 0)

    mask = config.mask(PerturbationType.SKIP_A2V_CROSS_ATTN, 0)
    assert mask.shape == (2, 1, 1)
    assert mask.data_ptr() != config.block_masks.data_ptr()
    sliced = config.batch_slice(1, 2)
    assert sliced.block_masks.shape == (4, 2, 1)
    assert sliced.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, 0)


def test_ltx_model_without_perturbations_matches_explicit_all_keep_masks() -> None:
    torch.manual_seed(0)
    model = _small_model(LTXModelType.VideoOnly).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    grid = torch.tensor([[0, h, w] for h in range(2) for w in range(2)], dtype=torch.float32).T
    video = Modality(
        latent=torch.randn(1, 4, 4),
        sigma=torch.tensor([0.5]),
        timesteps=torch.full((1, 4), 0.5),
        positions=torch.stack([grid, grid + 1], dim=-1)[None],
        context=torch.randn(1, 3, 16),
    )

    with torch.no_grad():
        implicit, _ = model(video, None, None)
        explicit, _ = model(video, None, BatchedPerturbationConfig.empty(1, model.num_blocks, dtype=torch.float32))

    torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)


def _word_tokenizer(*, adds_bos: bool) -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "red": 3, "fox": 4}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    if adds_bos:
        backend.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")


@pytest.mark.parametrize("adds_bos", [True, False], ids=["gemma3-style", "gemma4-style"])
def test_gemma_tokenizer_emits_exactly_one_leading_bos(adds_bos: bool) -> None:
    tokenizer = LTXVGemmaTokenizer(_word_tokenizer(adds_bos=adds_bos), max_length=6)

    pairs = [(int(token), int(mask)) for token, mask in tokenizer.tokenize_with_weights("  red fox ")["gemma"]]

    assert pairs == [(0, 0), (0, 0), (0, 0), (2, 1), (3, 1), (4, 1)]


def test_tile_size_config_geometry_on_the_latent_grid() -> None:
    scale = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    config = TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=16, overlap=8),
        height=DimensionSizeConfig(tile_size=64, overlap=32),
        width=DimensionSizeConfig(tile_size=64, overlap=32),
    )

    frames, height, width = config.to_splitters(scale)

    assert [(i.start, i.end, i.left_ramp, i.right_ramp) for i in height(4).intervals] == [
        (0, 2, 0, 1),
        (1, 3, 1, 1),
        (2, 4, 1, 0),
    ]
    assert [(i.start, i.end) for i in frames(5).intervals] == [(0, 2), (0, 3), (1, 4), (2, 5)]
    assert width(2).intervals[0].end == 2
    assert config.video_chunks_number(17) == len(frames(3).intervals) == 2
    with pytest.raises(ValueError, match="Overlap must be less than tile size"):
        DimensionSizeConfig(tile_size=8, overlap=8)


def test_split_by_size_grows_a_short_last_tile_to_the_minimum() -> None:
    intervals = split_by_size(4, 1, min_tile_size=3)(8).intervals

    assert [(i.start, i.end, i.left_ramp, i.right_ramp) for i in intervals] == [
        (0, 4, 0, 1),
        (3, 7, 1, 2),
        (5, 8, 2, 0),
    ]
    causal = split_temporal_causal(4, 1)(8).intervals
    assert [(i.start, i.left_ramp) for i in causal] == [(0, 0), (2, 2), (5, 2)]


def _transformer_metadata(**updates) -> dict:
    transformer = {
        "dropout": 0.0,
        "attention_bias": True,
        "num_vector_embeds": None,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 1000,
        "use_linear_projection": False,
        "only_cross_attention": False,
        "cross_attention_norm": True,
        "double_self_attention": False,
        "upcast_attention": False,
        "standardization_norm": "rms_norm",
        "norm_elementwise_affine": False,
        "qk_norm": "rms_norm",
        "positional_embedding_type": "rope",
        "use_audio_video_cross_attention": True,
        "share_ff": False,
        "av_cross_ada_norm": True,
        "use_middle_indices_grid": True,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 4,
        "in_channels": 4,
        "out_channels": 4,
        "audio_in_channels": 4,
        "audio_out_channels": 4,
        "num_layers": 1,
        "cross_attention_dim": 16,
        "audio_cross_attention_dim": 8,
        "caption_proj_before_connector": True,
        "cross_attention_adaln": True,
        "apply_gated_attention": True,
        "connector_num_attention_heads": 2,
        "connector_attention_head_dim": 8,
        "connector_num_layers": 1,
        "audio_connector_num_attention_heads": 2,
        "audio_connector_attention_head_dim": 4,
    }
    transformer.update(updates)
    return {"config": {"transformer": transformer}}


def test_transformer_from_metadata_reads_ltx25_flags_and_pins_eager_adaln() -> None:
    metadata = _transformer_metadata(ff_bias=False, use_keyframes_abs_pos_embedding=True)
    with torch.device("meta"):
        model = LTXModelConfigurator.from_metadata(metadata)
        reference = LTXModelConfigurator.from_config(metadata["config"])

    assert _ff_biases(model) == (False, True)
    assert model.keyframes_abs_pos_embedding is not None
    assert isinstance(model.transformer_blocks[0].ada_zero_function, PytorchAdaZeroFunction)
    assert model.transformer_blocks[0].ada_zero_function.backend == "torch"
    assert reference.transformer_blocks[0].ada_zero_function is DEFAULT_TRANSFORMER_OPS.ada_zero_function


@pytest.mark.parametrize("ff_bias", [True, False])
def test_connectors_follow_connector_ff_bias_and_audio_widths(ff_bias: bool) -> None:
    metadata = _transformer_metadata(connector_ff_bias=ff_bias)
    video = Embeddings1DConnectorConfigurator.from_metadata(metadata)
    audio = AudioEmbeddings1DConnectorConfigurator.from_metadata(metadata)

    assert video.inner_dim == 16
    assert audio.inner_dim == 8
    for connector in (video, audio):
        feed_forward = connector.transformer_1d_blocks[0].ff
        assert isinstance(feed_forward, torch.nn.Module)
        linears = [m for m in feed_forward.modules() if isinstance(m, torch.nn.Linear)]
        assert all((m.bias is not None) is ff_bias for m in linears)


def test_connector_ff_bias_defaults_to_biased_layers() -> None:
    connector = Embeddings1DConnectorConfigurator.from_config(_transformer_metadata()["config"])
    feed_forward = connector.transformer_1d_blocks[0].ff
    assert isinstance(feed_forward, torch.nn.Module)
    linears = [m for m in feed_forward.modules() if isinstance(m, torch.nn.Linear)]

    assert linears
    assert all(m.bias is not None for m in linears)


def test_split_checkpoint_component_configurators_read_synthetic_metadata() -> None:
    vae = {
        "config": {
            "vae": {
                "_class_name": "CausalDiffusionVAE",
                "encoder": {"in_channels": 3, "out_channels": 8, "patch_size": 4, "blocks": [["compress_all", {}]]},
            }
        }
    }
    with torch.device("meta"):
        encoder = VideoEncoderConfigurator.from_metadata(vae)
        upsampler = LatentUpsamplerConfigurator.from_metadata(
            {"config": {"in_channels": 8, "mid_channels": 32, "num_blocks_per_stage": 1}}
        )
        head = DurationHeadConfigurator.from_metadata(
            {
                "config": {
                    "transformer": {"cross_attention_dim": 16, "audio_cross_attention_dim": 8},
                    "duration_head": {"pooler_hidden_dim": 8, "num_queries": 2, "num_pooler_heads": 2, "mlp_hidden": 4},
                }
            }
        )

    assert encoder.latent_channels == 8
    assert encoder.patch_size == 4
    assert upsampler.state_dict()["initial_conv.weight"].shape[:2] == (32, 8)
    assert head.attention_pooler.query_tokens.shape == (2, 8)
    assert head.video_input_proj.in_features == 16
    assert head.audio_input_proj.in_features == 8


def test_vocoder_configurator_builds_bandwidth_extension_from_nested_config() -> None:
    stage = {
        "upsample_initial_channel": 16,
        "resblock": "AMP1",
        "upsample_rates": [2, 2],
        "resblock_kernel_sizes": [3],
        "upsample_kernel_sizes": [4, 4],
        "resblock_dilation_sizes": [[1, 3, 5]],
        "stereo": True,
        "use_tanh_at_final": False,
        "activation": "snakebeta",
        "use_bias_at_final": False,
    }
    bwe = {
        **stage,
        "upsample_initial_channel": 8,
        "apply_final_activation": False,
        "input_sampling_rate": 16000,
        "output_sampling_rate": 48000,
        "hop_length": 80,
        "n_fft": 512,
        "win_size": 512,
        "num_mels": 64,
    }
    with torch.device("meta"):
        vocoder = VocoderConfigurator.from_metadata({"config": {"vocoder": {"vocoder": stage, "bwe": bwe}}})

    assert isinstance(vocoder, VocoderWithBWE)
    assert vocoder.output_sampling_rate == 48000


def test_diffusion_video_decoder_configurator_reads_nested_decoder_config() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_video_decoder import (
        DiffusionVideoDecoderConfigurator,
    )

    decoder_config = {
        "in_channels": 16,
        "head_dim": 16,
        "stage_channels": [128, 64, 32, 32, 16],
        "stage_depths": [1, 1, 1, 1, 2],
        "t_emb_dim": 16,
    }
    with torch.device("meta"):
        decoder = DiffusionVideoDecoderConfigurator.from_metadata(
            {"config": {"vae": {"_class_name": "CausalDiffusionVAE", "decoder": decoder_config}}}
        )

    assert decoder.stage_channels == (128, 64, 32, 32, 16)
    assert len(decoder.diff_blocks) == 2
    assert decoder.conv_in.weight.shape[1] == 16
    assert decoder.type_emb.shape == (16,)
