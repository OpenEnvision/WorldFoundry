from __future__ import annotations

from worldfoundry.training import post_training
from worldfoundry.training.post_training import rl
from worldfoundry.training.post_training.rl import algorithms
from worldfoundry.training.post_training.rl.algorithms import diffusion_nft


def test_diffusion_nft_lazy_facades_resolve_to_canonical_runtime() -> None:
    names = (
        "DiffusionNFTRollout",
        "DiffusionNFTRewardAdapter",
        "DiffusionNFTTerminalLatents",
        "OldPolicyRefresh",
        "NativeDiffusionNFTEngine",
        "NativeDiffusionNFTTerminalCollector",
        "NativeDiffusionNFTTrainingStack",
        "NativeDiffusionNFTTrainingSession",
        "build_native_diffusion_nft_training_stack",
        "diffusion_nft_forward_process",
        "diffusion_nft_loss",
        "diffusion_nft_reward_weights",
    )
    for name in names:
        canonical = getattr(diffusion_nft, name)
        assert getattr(algorithms, name) is canonical
        assert getattr(rl, name) is canonical
        assert getattr(post_training, name) is canonical
