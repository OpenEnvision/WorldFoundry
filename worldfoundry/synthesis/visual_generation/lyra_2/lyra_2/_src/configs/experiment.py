from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


def register_lyra2():
    """Fully-flattened lyra_2 spatial runtime experiment.

    Effective config equivalent to:
    two_buffers_dl3dv_image_tokens_correspondence_finetune_kq_only_multibuffer_add_depth_hsg
    in the source repo.
    """
    experiment_config = {
        "defaults": [
            {"override /model": "fsdp_wan2pt1_lyra2_spatial"},
            {"override /net": "wan2pt1_14B_i2v_lyra2"},
            {"override /conditioner": "lyra2_conditioner"},
            "_self_",
        ],
        "job": {
            "project": "lyra_2",
            "group": "lyra2",
            "name": "lyra2",
        },
        "model": {
            "config": {
                "framepack_type": "f1k1f4s2f1s1f16k4f2k2f1k1_g20",
                "max_segments": 13,
                "apply_corruption_to_spatial_region": "noise_with_sigma",
                "augment_sigma_sample_p_mean": -3.0,
                "augment_sigma_sample_p_std": 2.0,
                "augment_sigma_sample_multiplier": 1.0,
                "self_aug_enabled": True,
                "self_aug_steps": 1,
                "self_aug_guidance": 1.0,
                "self_aug_scheduler_shift": 1.0,
                "self_aug_every_k": 2,
                "self_aug_prob": 1.0,
                "self_aug_max_T": 500,
                "self_aug_copy_chunk": True,
                "self_aug_encode_gt_with_clean_history": True,
                "starting_frame_ratio": 0.0,
                "use_mp_policy_fsdp": True,
                "keep_original_net_dtype": True,
                "spatial_memory_use_image": True,
                "spatial_memory_stride": 8,
                "spatial_memory_skip_recent": 16,
                "warp_chunk_size": 16,
                "framepack_trainable_modules": "cam_encoder,buffer_encoder,self_attn,clean_patch_embeddings,patch_embedding",
            },
        },
        "model_parallel": {
            "context_parallel_size": 1,
        },
        "checkpoint": {
            "load_from_object_store": {"enabled": False},
            "load_path": "",
            "strict_resume": False,
        },
    }

    cs.store(
        group="experiment",
        package="_global_",
        name="lyra2",
        node=experiment_config,
    )


register_lyra2()
