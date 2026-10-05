"""Native recipes for released Matrix Game 3.5 first/third-person checkpoints.

Each public function returns a :class:`NativeDiffusionRecipe`.  Recipes
do not construct runners.

Both viewpoints bind the Matrix-Game 3.5 DiT (subject-reference memory
capacity differs), Wan UMT5 conditioner (input passthrough), Wan T2V
initializer (48-ch / 16x / 4x, matching Wan 2.2 VAE 3.8), Wan
flow-match Euler (shift 5.0), and the Wan 2.2 VAE38 decoder.  Execution
is the default ``standard`` five-role binding.  CFG is on.  Family
options: camera-conditioned video, mosaic memory, ProPE, and subject
reference memory (2 refs first-person, 4 third-person).
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.wan import build_wan_video_vae38_decoder
from ..models.denoisers.matrix_game_3p5 import build_matrix_game_35_denoiser
from ..models.encoders.wan import build_wan_text_conditioner
from ..models.initializers.wan import build_wan_t2v_latent_initializer
from ..schedulers import build_wan_flow_match_euler_scheduler
from .spec import NativeDiffusionRecipe
from .wan import WAN_TOKENIZER_FILES

MATRIX_GAME_35_CHECKPOINT_REPO = "RiemannDynamics/Matrix-Game-3.5-Base"
MATRIX_GAME_35_CHECKPOINT_REVISION = "40c172355efa32d4e2a44076569e310807788f8a"
WAN22_TI2V_5B_REPO = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_TI2V_5B_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"

MATRIX_GAME_35_FIRST_PERSON_MODEL_ID = "matrix-game-3.5-first-person"
MATRIX_GAME_35_THIRD_PERSON_MODEL_ID = "matrix-game-3.5-third-person"


def _matrix_game_35_recipe(
    *,
    model_id: str,
    checkpoint_filename: str,
    viewpoint: str,
    subject_ref_capacity: int,
) -> NativeDiffusionRecipe:
    """Bind Matrix-Game 3.5 DiT, Wan text/VAE38, Euler flow-match, ``standard``."""

    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=MATRIX_GAME_35_CHECKPOINT_REPO,
            revision=MATRIX_GAME_35_CHECKPOINT_REVISION,
            files=(checkpoint_filename,),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO,
            revision=WAN22_TI2V_5B_REVISION,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO,
            revision=WAN22_TI2V_5B_REVISION,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO,
            revision=WAN22_TI2V_5B_REVISION,
            files=("Wan2.2_VAE.pth",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=(f"{MATRIX_GAME_35_CHECKPOINT_REPO}:{viewpoint}",),
        components=(
            ComponentSpec(
                key=ComponentKey(ComponentKind.DENOISER),
                factory=build_matrix_game_35_denoiser,
                checkpoints={"weights": "dit"},
                options={"subject_ref_memory_max_refs": subject_ref_capacity},
            ),
            ComponentSpec(
                key=ComponentKey(ComponentKind.CONDITIONER),
                factory=build_wan_text_conditioner,
                checkpoints={
                    "weights": "text-encoder",
                    "tokenizer": "tokenizer",
                },
                options={"passthrough_inputs": True},
            ),
            ComponentSpec(
                key=ComponentKey(ComponentKind.LATENT_INITIALIZER),
                factory=build_wan_t2v_latent_initializer,
                options={
                    "channels": 48,
                    "spatial_compression": 16,
                    "temporal_compression": 4,
                },
            ),
            ComponentSpec(
                key=ComponentKey(ComponentKind.SCHEDULER),
                factory=build_wan_flow_match_euler_scheduler,
                options={"shift": 5.0},
            ),
            ComponentSpec(
                key=ComponentKey(ComponentKind.DECODER),
                factory=build_wan_video_vae38_decoder,
                checkpoints={"weights": "vae"},
            ),
        ),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "camera-conditioned-video",
                "classifier-free-guidance",
                "mosaic-memory",
                "prope",
                "subject-reference-memory",
            }
        ),
        options={
            "latent_channels": 48,
            "spatial_compression": 16,
            "temporal_compression": 4,
            "initial_latent_distribution": "gaussian",
        },
        metadata={
            "architecture": "matrix-game-3.5",
            "viewpoint": viewpoint,
            "subject_ref_capacity": subject_ref_capacity,
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


def matrix_game_35_first_person_recipe() -> NativeDiffusionRecipe:
    """First-person 3.5 checkpoint; subject-ref capacity 2, ``standard`` CFG."""

    return _matrix_game_35_recipe(
        model_id=MATRIX_GAME_35_FIRST_PERSON_MODEL_ID,
        checkpoint_filename="first-person.safetensors",
        viewpoint="first-person",
        subject_ref_capacity=2,
    )


def matrix_game_35_third_person_recipe() -> NativeDiffusionRecipe:
    """Third-person 3.5 checkpoint; subject-ref capacity 4, ``standard`` CFG."""

    return _matrix_game_35_recipe(
        model_id=MATRIX_GAME_35_THIRD_PERSON_MODEL_ID,
        checkpoint_filename="third-person.safetensors",
        viewpoint="third-person",
        subject_ref_capacity=4,
    )


__all__ = [
    "MATRIX_GAME_35_CHECKPOINT_REPO",
    "MATRIX_GAME_35_CHECKPOINT_REVISION",
    "MATRIX_GAME_35_FIRST_PERSON_MODEL_ID",
    "MATRIX_GAME_35_THIRD_PERSON_MODEL_ID",
    "WAN22_TI2V_5B_REPO",
    "WAN22_TI2V_5B_REVISION",
    "matrix_game_35_first_person_recipe",
    "matrix_game_35_third_person_recipe",
]
