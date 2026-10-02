from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_action_generation_is_not_a_base_models_domain() -> None:
    assert not (REPO_ROOT / "worldfoundry/base_models/action_generation").exists()


def test_gr00t_runtime_lives_in_synthesis_package() -> None:
    synthesis_runtime = REPO_ROOT / "worldfoundry/synthesis/action_generation/gr00t/runtime.py"
    synthesis_wrapper = REPO_ROOT / "worldfoundry/synthesis/action_generation/gr00t/gr00t_synthesis.py"

    assert synthesis_runtime.is_file()
    assert "worldfoundry.synthesis.action_generation.gr00t.runtime" in _imports_from(synthesis_wrapper)


def test_gr00t_runtime_does_not_keep_training_style_config_package() -> None:
    runtime_root = REPO_ROOT / "worldfoundry/synthesis/action_generation/gr00t/gr00t_runtime"
    checked_files = [
        runtime_root / "inference_support/utils.py",
        runtime_root / "inference_support/state_action/state_action_processor.py",
        runtime_root / "policy/gr00t_policy.py",
        runtime_root / "model/gr00t_n1d7/gr00t_n1d7.py",
        runtime_root / "model/gr00t_n1d7/processing_gr00t_n1d7.py",
    ]

    assert not (runtime_root / "configs").exists()
    assert not (runtime_root / "data").exists()
    assert (runtime_root / "inference_support/embodiment_configs.py").is_file()
    assert (runtime_root / "model/gr00t_n1d7/configuration_gr00t_n1d7.py").is_file()
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert "gr00t.configs" not in text
        assert "gr00t.data" not in text
    assert "import gr00t.model  # noqa: F401" not in (runtime_root / "policy/gr00t_policy.py").read_text(
        encoding="utf-8"
    )


def test_octo_runtime_does_not_package_training_state_helpers() -> None:
    octo_root = REPO_ROOT / "worldfoundry/synthesis/action_generation/octo/octo_runtime/octo"
    model_path = octo_root / "model/octo_model.py"

    assert (octo_root / "model/octo_model.py").is_file()
    assert not (octo_root / "utils/train_callbacks.py").exists()
    assert not (octo_root / "utils/train_utils.py").exists()

    model_text = model_path.read_text(encoding="utf-8")
    assert "octo.utils.train_utils" not in model_text
    assert "TrainState.create" not in model_text


def test_openpi_runtime_support_is_inference_only() -> None:
    openpi_root = REPO_ROOT / "worldfoundry/synthesis/action_generation/openpi/openpi_runtime/openpi"
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/action_generation/openpi/runtime.py"
    policy_config_path = openpi_root / "policies/policy_config.py"
    runtime_support = openpi_root / "runtime_support"

    assert not (openpi_root / "training").exists()
    assert (runtime_support / "config.py").is_file()
    assert (runtime_support / "checkpoints.py").is_file()
    assert (runtime_support / "sharding.py").is_file()
    assert (runtime_support / "weight_loaders.py").is_file()

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            runtime_path,
            policy_config_path,
            openpi_root / "models/gemma.py",
            openpi_root / "models/siglip.py",
        )
    )
    legacy_training_import = ".".join(("openpi", "training"))
    assert legacy_training_import not in combined
    assert "openpi.runtime_support" in combined


def test_being_h05_runtime_support_is_inference_named() -> None:
    beingh_root = REPO_ROOT / "worldfoundry/synthesis/action_generation/being_h05/being_h05_runtime/BeingH"
    runtime_path = REPO_ROOT / "worldfoundry/synthesis/action_generation/being_h05/runtime.py"
    data_config_path = beingh_root / "inference_support/data_config.py"
    policy_path = beingh_root / "inference/beingh_policy.py"
    inference_support = beingh_root / "inference_support"
    constants_path = beingh_root / "utils/constants.py"
    model_init_path = beingh_root / "model/__init__.py"
    state_action_path = inference_support / "transforms/state_action.py"

    assert not (beingh_root / "dataset").exists()
    assert not (REPO_ROOT / "worldfoundry/synthesis/action_generation/being_h05/being_h05_runtime/configs").exists()
    assert (inference_support / "image_transforms.py").is_file()
    assert (inference_support / "data_config.py").is_file()
    assert (inference_support / "transforms/base.py").is_file()
    assert (inference_support / "transforms/concat.py").is_file()
    assert (inference_support / "transforms/state_action.py").is_file()

    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (data_config_path, policy_path, runtime_path)
    )
    legacy_dataset_import = ".".join(("BeingH", "dataset"))
    assert legacy_dataset_import not in combined
    assert "BeingH.inference_support" in combined
    assert "configs.data_config" not in combined

    constants_text = constants_path.read_text(encoding="utf-8")
    model_init_text = model_init_path.read_text(encoding="utf-8")
    state_action_text = state_action_path.read_text(encoding="utf-8")
    assert "from BeingH.model" not in constants_text
    assert "_LazyArchRegistry" in constants_text
    assert "__getattr__" in model_init_text
    assert "import pytorch3d.transforms as pt" not in state_action_text
    assert "def _pytorch3d_transforms" in state_action_text


def test_action_runtime_wrappers_use_synthesis_package() -> None:
    runtime_wrappers = {
        "being_h05": "being_h05_synthesis.py",
        "diffusion_policy": "diffusion_policy_synthesis.py",
        "dreamzero": "dreamzero_synthesis.py",
        "giga_brain_0": "giga_brain_0_synthesis.py",
        "lapa": "lapa_synthesis.py",
        "lingbot_va": "lingbot_va_synthesis.py",
        "molmoact2": "molmoact2_synthesis.py",
        "openpi": "openpi_synthesis.py",
        "starvla": "starvla_synthesis.py",
    }
    for package, wrapper_name in runtime_wrappers.items():
        synthesis_runtime = REPO_ROOT / f"worldfoundry/synthesis/action_generation/{package}/runtime.py"
        synthesis_wrapper = REPO_ROOT / f"worldfoundry/synthesis/action_generation/{package}/{wrapper_name}"

        assert synthesis_runtime.is_file(), package
        assert f"worldfoundry.synthesis.action_generation.{package}.runtime" in _imports_from(synthesis_wrapper), package


def test_action_generation_runtime_configs_live_under_data_root() -> None:
    action_root = REPO_ROOT / "worldfoundry/synthesis/action_generation"
    remaining_configs = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in action_root.rglob("*")
        if path.suffix in {".json", ".yaml", ".yml"}
    )
    assert remaining_configs == []

    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs"
    expected_configs = [
        config_root / "vla_va_wam/act.yaml",
        config_root / "vla_va_wam/being-h05.yaml",
        config_root / "vla_va_wam/dreamzero.yaml",
        config_root / "vla_va_wam/giga-brain-0.yaml",
        config_root / "vla_va_wam/lapa.yaml",
        config_root / "vla_va_wam/lingbot-va.yaml",
        config_root / "vla_va_wam/octo.yaml",
        config_root / "vla_va_wam/openpi.yaml",
        config_root / "vla_va_wam/policy_algorithm_sources.yaml",
        config_root / "vla_va_wam/openvla.yaml",
        config_root / "rt1/film_efficientnet/imagenet_classes.yaml",
        config_root / "vla_va_wam/gr00t.yaml",
        config_root / "vla_va_wam/molmoact2.yaml",
        config_root / "vla_va_wam/starvla.yaml",
    ]
    for config_path in expected_configs:
        assert config_path.is_file(), config_path
    assert not (config_root / "lerobot").exists()

    assert not (config_root / "being_h05").exists()
    assert not (config_root / "dreamzero").exists()
    assert not (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/dreamzero/dreamzero_runtime/groot/vla/experiment"
    ).exists()
    assert not (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/rt1/rt1_runtime/robotics_transformer/configs"
    ).exists()
    assert not (REPO_ROOT / "worldfoundry/synthesis/action_generation/lerobot").exists()


def test_native_action_policy_source_resolver_prefers_in_tree_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from worldfoundry.synthesis.action_generation._native_policy_runtime import resolve_source_workdir

    explicit = tmp_path / "explicit-cogact"
    explicit.mkdir()
    env_checkout = tmp_path / "env-cogact"
    env_checkout.mkdir()
    sibling_checkout = tmp_path / "github_repos" / "cogact"
    sibling_checkout.mkdir(parents=True)
    model_source_checkout = tmp_path / "model_sources" / "cogact"
    model_source_checkout.mkdir(parents=True)

    monkeypatch.setenv("WORLDFOUNDRY_COGACT_REPO", str(env_checkout))
    monkeypatch.setenv("WORLDFOUNDRY_GITHUB_REPOS_ROOT", str(tmp_path / "github_repos"))
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "model_sources"))

    in_tree_subdir = "worldfoundry/synthesis/action_generation/cogact/cogact_runtime"
    assert resolve_source_workdir({"source_repo": str(explicit)}, "cogact", in_tree_subdir=in_tree_subdir) == explicit
    assert resolve_source_workdir(
        {},
        "cogact",
        specific_env="WORLDFOUNDRY_COGACT_REPO",
        in_tree_subdir=in_tree_subdir,
    ) == (REPO_ROOT / in_tree_subdir).resolve()


def test_native_action_policy_source_resolver_does_not_fallback_to_external_repos(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from worldfoundry.synthesis.action_generation._native_policy_runtime import resolve_source_workdir

    monkeypatch.setenv("WORLDFOUNDRY_GITHUB_REPOS_ROOT", str(tmp_path / "github_repos"))
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "model_sources"))

    resolved = resolve_source_workdir(
        {},
        "missing-action-runtime",
        specific_env="WORLDFOUNDRY_MISSING_ACTION_REPO",
        in_tree_subdir="worldfoundry/synthesis/action_generation/missing_action_runtime",
    )

    assert "github_repos" not in str(resolved)
    assert "model_sources" not in str(resolved)
    assert resolved == (REPO_ROOT / "worldfoundry/synthesis/action_generation/missing_action_runtime").resolve()


def test_mme_vla_history_configs_live_under_data_root() -> None:
    runtime_config_dir = (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/mme_vla/mme_vla_runtime/src/mme_vla_suite/models/config"
    )
    data_config_dir = (
        REPO_ROOT
        / "worldfoundry/data/models/runtime/configs/vla_va_wam/mme_vla/models/config"
    )

    assert (runtime_config_dir / "utils.py").is_file()
    assert not list(runtime_config_dir.rglob("*.yaml"))
    assert (data_config_dir / "base.yaml").is_file()
    assert (data_config_dir / "robomme/perceptual-framesamp-context.yaml").is_file()

    loader_text = (runtime_config_dir / "utils.py").read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_MME_VLA_CONFIG_ROOT" in loader_text
    for token in ("package_root", '"data"', '"runtime"', '"configs"', '"vla_va_wam"', '"mme_vla"', '"robomme"'):
        assert token in loader_text
    assert "src/mme_vla_suite/models/config/robomme" not in loader_text


def test_lerobot_policy_sources_are_tracked_as_concrete_algorithms() -> None:
    import yaml

    source_manifest = (
        REPO_ROOT / "worldfoundry/data/models/runtime/configs/vla_va_wam/policy_algorithm_sources.yaml"
    )
    payload = yaml.safe_load(source_manifest.read_text(encoding="utf-8"))
    policy_dirs = {
        path.name
        for path in (
            REPO_ROOT.parent / "github_repos/lerobot/src/lerobot/policies"
        ).iterdir()
        if path.is_dir()
    }
    manifest_dirs = {item["lerobot_policy_dir"] for item in payload["algorithms"]}

    assert policy_dirs == manifest_dirs
    assert "lerobot-policy-zoo" not in {item["worldfoundry_model_id"] for item in payload["algorithms"]}
    assert {item["source_basis"] for item in payload["algorithms"]} >= {
        "official_repo",
        "lerobot_policy_reference",
        "lerobot_reference_implementation",
    }


def test_action_runtime_code_resolves_external_configs() -> None:
    rt1_efficientnet = (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/rt1/rt1_runtime/robotics_transformer/"
        "film_efficientnet/film_efficientnet_encoder.py"
    ).read_text(encoding="utf-8")
    gr00t_synthesis = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/gr00t/gr00t_synthesis.py"
    ).read_text(encoding="utf-8")
    gr00t_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/gr00t/runtime.py"
    ).read_text(encoding="utf-8")
    being_h05_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/being_h05/runtime.py"
    ).read_text(encoding="utf-8")
    lapa_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/lapa/runtime.py"
    ).read_text(encoding="utf-8")
    openpi_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/openpi/runtime.py"
    ).read_text(encoding="utf-8")
    starvla_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/starvla/runtime.py"
    ).read_text(encoding="utf-8")
    molmoact2_synthesis = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/molmoact2/molmoact2_synthesis.py"
    ).read_text(encoding="utf-8")
    dreamzero_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/dreamzero/runtime.py"
    ).read_text(encoding="utf-8")
    giga_brain_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/giga_brain_0/runtime.py"
    ).read_text(encoding="utf-8")
    lingbot_runtime = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/lingbot_va/runtime.py"
    ).read_text(encoding="utf-8")
    external_config_wrappers = {
        "act/act_synthesis.py": ['"head_cam"', '"resnet18"', '"sine"'],
        "being_h05/being_h05_synthesis.py": ['"libero_nonorm"', '"libero_posttrain"', '"{task_description}"', '"dense"'],
        "dreamzero/dreamzero_synthesis.py": ['"0.0.0.0"', "DEFAULT_DREAMZERO_SERVER_PORT"],
        "lapa/lapa_synthesis.py": ['"bf16"', '"1,-1,1,1"'],
        "lingbot_va/lingbot_va_synthesis.py": ['"libero"', '"0.0.0.0"', "29536"],
        "octo/octo_synthesis.py": ['"small"', '"bridge_dataset"'],
        "openpi/openpi_synthesis.py": ['"pi05_libero"'],
        "openvla/openvla_synthesis.py": ['"bridge_orig"', '"auto"', '"eager"'],
        "starvla/starvla_synthesis.py": ['"vla_policy"', '"DiT-B"', '"sdpa"'],
    }
    assert 'worldfoundry_data_path(' in rt1_efficientnet
    assert '"rt1"' in rt1_efficientnet
    assert '"imagenet_classes.yaml"' in rt1_efficientnet
    assert "yaml.safe_load" in rt1_efficientnet
    assert "json.load" not in rt1_efficientnet
    assert "os.path.join(os.path.dirname(__file__), IMAGENET_JSON_PATH)" not in rt1_efficientnet

    assert "load_vla_va_wam_runtime_config" in gr00t_synthesis
    assert "load_vla_va_wam_runtime_config" in molmoact2_synthesis
    assert "load_vla_va_wam_runtime_config" in dreamzero_runtime
    assert "DEFAULT_" not in dreamzero_runtime
    assert "CAMERA_FILES" not in dreamzero_runtime
    assert "RELATIVE_OFFSETS" not in dreamzero_runtime
    assert "ACTION_HORIZON" not in dreamzero_runtime
    assert "DEFAULT_PORT" not in lingbot_runtime
    assert "CONFIG_BY_ROLE" not in lingbot_runtime
    assert "embodiment_tag: str = " not in gr00t_runtime
    assert "torch_dtype: str = " not in gr00t_runtime
    assert "dtype: str = " not in lapa_runtime
    assert "mesh_dim: str = " not in lapa_runtime
    assert "config_name: str = " not in openpi_runtime
    assert "data_config_name: str = " not in being_h05_runtime
    assert "attention_mask_kind: str = " not in being_h05_runtime
    assert "base_vlm: str = " not in starvla_runtime
    assert "playground/Pretrained_models/Qwen3-VL-4B-Instruct" not in starvla_runtime
    assert "action_chunk: int = " not in giga_brain_runtime
    assert "compile_policy: bool = " not in giga_brain_runtime
    assert "google/paligemma2-3b-pt-224" not in giga_brain_runtime
    assert "physical-intelligence/fast" not in giga_brain_runtime
    assert '"LIBERO_PANDA"' not in gr00t_synthesis
    assert '"allenai/MolmoAct2-DROID"' not in molmoact2_synthesis
    assert "EMBODIMENT_DEFAULTS" not in molmoact2_synthesis
    assert "Move the pan forward" not in dreamzero_runtime
    assert '"exterior_image_1_left.mp4"' not in dreamzero_runtime
    for relative_path, forbidden_literals in external_config_wrappers.items():
        text = (REPO_ROOT / "worldfoundry/synthesis/action_generation" / relative_path).read_text(encoding="utf-8")
        assert "load_vla_va_wam_runtime_config" in text
        for literal in forbidden_literals:
            assert literal not in text


def test_gr00t_runtime_config_yaml_controls_defaults(tmp_path: Path) -> None:
    import yaml

    from worldfoundry.synthesis.action_generation.gr00t import GR00TSynthesis

    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "custom").mkdir(parents=True)
    config_path = tmp_path / "gr00t.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_variant": "custom",
                "torch_dtype": "float16",
                "seed": 9,
                "variants": {"custom": {"embodiment_tag": "CUSTOM_ROBOT"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    synthesis = GR00TSynthesis.from_pretrained(
        model_id="gr00t",
        device="cpu",
        checkpoint_dir=checkpoint_root,
        runtime_config_path=config_path,
    )
    runtime_config = synthesis._runtime_config({})

    assert runtime_config.checkpoint_dir == (checkpoint_root / "custom").resolve()
    assert runtime_config.embodiment_tag == "CUSTOM_ROBOT"
    assert runtime_config.torch_dtype == "float16"
    assert runtime_config.seed == 9


def test_lightweight_vla_runtime_config_yamls_control_defaults(tmp_path: Path) -> None:
    import yaml

    from worldfoundry.synthesis.action_generation.lapa import LAPASynthesis
    from worldfoundry.synthesis.action_generation.octo import OctoSynthesis
    from worldfoundry.synthesis.action_generation.openpi import OpenPISynthesis
    from worldfoundry.synthesis.action_generation.openvla import OpenVLASynthesis

    openvla_checkpoint = tmp_path / "openvla-checkpoint"
    openvla_checkpoint.mkdir()
    openvla_config = tmp_path / "openvla.yaml"
    openvla_config.write_text(
        yaml.safe_dump(
            {
                "unnorm_key": "libero_10",
                "torch_dtype": "float16",
                "attn_implementation": "sdpa",
                "use_cache": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    openvla = OpenVLASynthesis.from_pretrained(
        model_id="openvla",
        device="cpu",
        checkpoint_dir=openvla_checkpoint,
        runtime_config_path=openvla_config,
    )._runtime_config({})
    assert openvla.checkpoint_dir == openvla_checkpoint.resolve()
    assert openvla.unnorm_key == "libero_10"
    assert openvla.torch_dtype == "float16"
    assert openvla.attn_implementation == "sdpa"
    assert openvla.use_cache is False

    openpi_checkpoint = tmp_path / "openpi-checkpoint"
    openpi_checkpoint.mkdir()
    openpi_config = tmp_path / "openpi.yaml"
    openpi_config.write_text(
        yaml.safe_dump({"config_name": "pi0_fast", "pytorch_device": "cpu", "seed": 11}, sort_keys=False),
        encoding="utf-8",
    )
    openpi = OpenPISynthesis.from_pretrained(
        model_id="openpi",
        device="cpu",
        checkpoint_dir=openpi_checkpoint,
        runtime_config_path=openpi_config,
    )._runtime_config({})
    assert openpi.checkpoint_dir == openpi_checkpoint.resolve()
    assert openpi.config_name == "pi0_fast"
    assert openpi.pytorch_device == "cpu"
    assert openpi.seed == 11

    octo_checkpoint = tmp_path / "octo-base"
    octo_checkpoint.mkdir()
    octo_config = tmp_path / "octo.yaml"
    octo_config.write_text(
        yaml.safe_dump(
            {
                "variant": "base",
                "dataset_key": "custom_bridge",
                "image_key": "rgb",
                "image_size": [128, 192],
                "jax_platform": "cpu",
                "seed": 22,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    octo = OctoSynthesis.from_pretrained(
        model_id="octo",
        device="cpu",
        checkpoint_dir=octo_checkpoint,
        runtime_config_path=octo_config,
    )._runtime_config({}, require_checkpoint=True)
    assert octo.checkpoint_dir == octo_checkpoint.resolve()
    assert octo.variant == "base"
    assert octo.dataset_key == "custom_bridge"
    assert octo.image_key == "rgb"
    assert octo.image_size == (128, 192)
    assert octo.seed == 22

    lapa_checkpoint = tmp_path / "lapa-checkpoint"
    lapa_checkpoint.mkdir()
    for name in ("params", "tokenizer.model", "vqgan"):
        (lapa_checkpoint / name).write_text("fixture", encoding="utf-8")
    lapa_config = tmp_path / "lapa.yaml"
    lapa_config.write_text(
        yaml.safe_dump(
            {
                "dtype": "fp16",
                "image_size": 384,
                "mesh_dim": "1,1,1,1",
                "seed": 33,
                "tokens_per_delta": 8,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    lapa = LAPASynthesis.from_pretrained(
        model_id="lapa",
        device="cpu",
        checkpoint_dir=lapa_checkpoint,
        runtime_config_path=lapa_config,
    )._runtime_config({})
    assert lapa.assets.checkpoint_dir == lapa_checkpoint.resolve()
    assert lapa.dtype == "fp16"
    assert lapa.image_size == 384
    assert lapa.mesh_dim == "1,1,1,1"
    assert lapa.seed == 33
    assert lapa.tokens_per_delta == 8


def test_action_model_runtime_config_yamls_control_defaults(tmp_path: Path) -> None:
    import yaml

    from worldfoundry.synthesis.action_generation.act import ACTSynthesis
    from worldfoundry.synthesis.action_generation.lingbot_va import LingBotVASynthesis
    from worldfoundry.synthesis.action_generation.lingbot_va.runtime import config_name_for_checkpoint
    from worldfoundry.synthesis.action_generation.starvla import StarVLASynthesis

    act_checkpoint = tmp_path / "policy_best.ckpt"
    act_checkpoint.write_bytes(b"fixture")
    act_config = tmp_path / "act.yaml"
    act_config.write_text(
        yaml.safe_dump(
            {
                "camera_names": ["front", "wrist"],
                "state_dim": 8,
                "chunk_size": 12,
                "temporal_agg": True,
                "lr": 2.0e-5,
                "lr_backbone": 3.0e-5,
                "weight_decay": 2.0e-4,
                "backbone": "resnet34",
                "dilation": True,
                "position_embedding": "learned",
                "enc_layers": 2,
                "dec_layers": 3,
                "dim_feedforward": 1024,
                "hidden_dim": 256,
                "dropout": 0.2,
                "nheads": 4,
                "pre_norm": True,
                "masks": True,
                "kl_weight": 5.0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    act = ACTSynthesis.from_pretrained(
        model_id="act",
        device="cpu",
        checkpoint_path=act_checkpoint,
        runtime_config_path=act_config,
    )._runtime_config({}, require_checkpoint=True)
    assert act is not None
    assert act.checkpoint_path == act_checkpoint.resolve()
    assert act.camera_names == ("front", "wrist")
    assert act.state_dim == 8
    assert act.chunk_size == 12
    assert act.temporal_agg is True
    assert act.backbone == "resnet34"
    assert act.position_embedding == "learned"
    assert act.nheads == 4
    assert act.kl_weight == 5.0

    starvla_checkpoint = tmp_path / "starvla-checkpoint"
    starvla_checkpoint.mkdir()
    starvla_source = tmp_path / "starvla-source"
    starvla_source.mkdir()
    starvla_config = tmp_path / "starvla.yaml"
    starvla_config.write_text(
        yaml.safe_dump(
            {
                "track": "world_action",
                "variant_id": "wm4a",
                "base_vlm": "custom/base-vlm",
                "action_model_type": "DiT-L",
                "action_dim": 9,
                "action_horizon": 6,
                "source_repo_dir": str(starvla_source),
                "attn_implementation": "eager",
                "enable_official_runtime": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    starvla = StarVLASynthesis.from_pretrained(
        model_id="starvla",
        device="cpu",
        checkpoint_dir=starvla_checkpoint,
        runtime_config_path=starvla_config,
    )._runtime_config({})
    assert starvla.checkpoint_dir == starvla_checkpoint.resolve()
    assert starvla.track == "world_action"
    assert starvla.base_vlm == "custom/base-vlm"
    assert starvla.action_model_type == "DiT-L"
    assert starvla.action_dim == 9
    assert starvla.action_horizon == 6
    assert starvla.source_repo_dir == starvla_source.resolve()
    assert starvla.attn_implementation == "eager"
    assert starvla.enable_official_runtime is True

    lingbot_checkpoint = tmp_path / "lingbot-checkpoint"
    lingbot_checkpoint.mkdir()
    lingbot_config = tmp_path / "lingbot-va.yaml"
    lingbot_config.write_text(
        yaml.safe_dump(
            {
                "default_config_name": "robotwin",
                "server_host": "127.0.0.1",
                "server_port": 29999,
                "nproc_per_node": 2,
                "master_port": 29998,
                "checkpoint_role_config_names": {
                    "robotwin_posttrain_demo_checkpoint": "robotwin",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    lingbot = LingBotVASynthesis.from_pretrained(
        model_id="lingbot-va",
        device="cpu",
        checkpoint_dir=lingbot_checkpoint,
        runtime_config_path=lingbot_config,
    )._runtime_config({}, require_checkpoint=True)
    assert lingbot.checkpoint_dir == lingbot_checkpoint.resolve()
    assert lingbot.config_name == "robotwin"
    assert lingbot.host == "127.0.0.1"
    assert lingbot.port == 29999
    assert lingbot.nproc_per_node == 2
    assert lingbot.master_port == 29998

    robotwin_checkpoint = tmp_path / "robotwin-checkpoint"
    robotwin_checkpoint.mkdir()
    assert (
        config_name_for_checkpoint(
            robotwin_checkpoint,
            [{"local_dir": str(robotwin_checkpoint), "role": "robotwin_posttrain_demo_checkpoint"}],
            config_by_role={"robotwin_posttrain_demo_checkpoint": "robotwin"},
            fallback="libero",
        )
        == "robotwin"
    )


def test_starvla_default_source_repo_is_in_tree(monkeypatch, tmp_path: Path) -> None:
    from worldfoundry.synthesis.action_generation.starvla import runtime as starvla_runtime

    monkeypatch.delenv("WORLDFOUNDRY_STARVLA_REPO_ROOT", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "model_sources"))
    monkeypatch.setenv("WORLDFOUNDRY_GITHUB_REPOS_ROOT", str(tmp_path / "github_repos"))

    source = Path(starvla_runtime.__file__).read_text(encoding="utf-8")
    assert "official_runtime_repo_path" not in source
    assert "github_repos" not in source
    assert "WORLDFOUNDRY_MODEL_SOURCE_DIR" not in source
    assert starvla_runtime._default_source_repo_dir() == starvla_runtime.RUNTIME_ROOT

    override = tmp_path / "starvla-override"
    monkeypatch.setenv("WORLDFOUNDRY_STARVLA_REPO_ROOT", str(override))
    assert starvla_runtime._default_source_repo_dir() == override


def test_remaining_action_runtime_config_yamls_control_defaults(tmp_path: Path) -> None:
    import yaml

    from worldfoundry.synthesis.action_generation.being_h05 import BeingH05Synthesis
    from worldfoundry.synthesis.action_generation.dreamzero import DreamZeroSynthesis
    from worldfoundry.synthesis.action_generation.giga_brain_0 import GigaBrain0Synthesis

    being_checkpoint = tmp_path / "being-h05-checkpoint"
    being_checkpoint.mkdir()
    being_config = tmp_path / "being-h05.yaml"
    being_config.write_text(
        yaml.safe_dump(
            {
                "data_config_name": "robocasa_nonorm",
                "dataset_name": "robocasa_posttrain",
                "embodiment_tag": "robocasa",
                "instruction_template": "Instruction: {task_description}",
                "enable_rtc": False,
                "metadata_variant": "robocasa",
                "stats_selection_mode": "checkpoint",
                "attention_mask_kind": "causal",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    being = BeingH05Synthesis.from_pretrained(
        model_id="being-h05",
        device="cpu",
        checkpoint_dir=being_checkpoint,
        runtime_config_path=being_config,
    )._runtime_config({})
    assert being.checkpoint_dir == being_checkpoint.resolve()
    assert being.data_config_name == "robocasa_nonorm"
    assert being.dataset_name == "robocasa_posttrain"
    assert being.embodiment_tag == "robocasa"
    assert being.instruction_template == "Instruction: {task_description}"
    assert being.enable_rtc is False
    assert being.metadata_variant == "robocasa"
    assert being.stats_selection_mode == "checkpoint"
    assert being.attention_mask_kind == "causal"

    dreamzero_checkpoint = tmp_path / "dreamzero-checkpoint"
    dreamzero_checkpoint.mkdir()
    dreamzero_config = tmp_path / "dreamzero.yaml"
    dreamzero_config.write_text(
        yaml.safe_dump(
            {
                "variant": "wan22",
                "server_host": "127.0.0.1",
                "server_port": 18000,
                "nproc_per_node": 1,
                "enable_dit_cache": False,
                "max_chunk_size": 3,
                "client_demo": {
                    "prompt": "custom instruction",
                    "camera_files": {
                        "observation/front": "front.mp4",
                        "observation/wrist": "wrist.mp4",
                    },
                    "relative_offsets": [-2, 0],
                    "action_horizon": 5,
                    "num_chunks": 4,
                    "zero_image_height": 64,
                    "zero_image_width": 128,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dreamzero = DreamZeroSynthesis.from_pretrained(
        model_id="dreamzero",
        device="cpu",
        checkpoint_dir=dreamzero_checkpoint,
        runtime_config_path=dreamzero_config,
    )._runtime_config({}, require_checkpoint=False)
    assert dreamzero.checkpoint_dir == dreamzero_checkpoint.resolve()
    assert dreamzero.host == "127.0.0.1"
    assert dreamzero.port == 18000
    assert dreamzero.nproc_per_node == 1
    assert dreamzero.enable_dit_cache is False
    assert dreamzero.max_chunk_size == 3
    assert dreamzero.client_demo["prompt"] == "custom instruction"
    assert dreamzero.client_demo["camera_files"] == {
        "observation/front": "front.mp4",
        "observation/wrist": "wrist.mp4",
    }
    assert dreamzero.client_demo["relative_offsets"] == [-2, 0]
    assert dreamzero.client_demo["action_horizon"] == 5
    assert dreamzero.client_demo["num_chunks"] == 4
    assert dreamzero.client_demo["zero_image_height"] == 64
    assert dreamzero.client_demo["zero_image_width"] == 128

    giga_model = tmp_path / "giga-brain-model"
    giga_model.mkdir()
    giga_stats = tmp_path / "giga-brain-stats.json"
    giga_stats.write_text("{}", encoding="utf-8")
    giga_config = tmp_path / "giga-brain-0.yaml"
    giga_config.write_text(
        yaml.safe_dump(
            {
                "action_chunk": 7,
                "compile_policy": False,
                "torch_dtype": "float16",
                "autoregressive_mode_only": True,
                "enable_2d_traj_output": True,
                "depth_img_prefix_name": "depth",
                "tokenizer_model_path": "custom/tokenizer-model",
                "fast_tokenizer_path": "custom/fast-tokenizer",
                "fallback_action_dim": 5,
                "fallback_embodiment_id": 4,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    giga_brain = GigaBrain0Synthesis.from_pretrained(
        model_id="giga-brain-0",
        device="cpu",
        model_path=giga_model,
        norm_stats_path=giga_stats,
        delta_mask="1,0,1",
        original_action_dim=3,
        embodiment_id=2,
        runtime_config_path=giga_config,
    )._runtime_config({}, require_existing=True)
    assert giga_brain.model_path == giga_model.resolve()
    assert giga_brain.norm_stats_path == giga_stats.resolve()
    assert giga_brain.delta_mask == (True, False, True)
    assert giga_brain.original_action_dim == 3
    assert giga_brain.embodiment_id == 2
    assert giga_brain.tokenizer_model_path == "custom/tokenizer-model"
    assert giga_brain.fast_tokenizer_path == "custom/fast-tokenizer"
    assert giga_brain.action_chunk == 7
    assert giga_brain.compile_policy is False
    assert giga_brain.torch_dtype == "float16"
    assert giga_brain.autoregressive_mode_only is True
    assert giga_brain.enable_2d_traj_output is True
    assert giga_brain.depth_img_prefix_name == "depth"


def test_dreamzero_wan_foundation_modules_live_under_base_models() -> None:
    synthesis_modules = (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/dreamzero/dreamzero_runtime/groot/vla/model/dreamzero/modules"
    )
    base_modules = (
        REPO_ROOT
        / "worldfoundry/base_models/diffusion_model/video/wan/wan_dreamzero/modules"
    )
    moved_modules = {
        "attention.py",
        "cudnn_attention.py",
        "flow_match_scheduler.py",
        "flow_unipc_multistep_scheduler.py",
        "utils.py",
        "vram_management.py",
        "wan2_1_attention.py",
        "wan2_1_submodule.py",
        "wan_video_camera_controller.py",
        "wan_video_dit.py",
        "wan_video_image_encoder.py",
        "wan_video_text_encoder.py",
        "wan_video_vae.py",
    }

    assert (base_modules / "__init__.py").is_file()
    assert (synthesis_modules / "wan_video_dit_action_casual_chunk.py").is_file()
    for module_name in moved_modules:
        base_path = base_modules / module_name
        shim_path = synthesis_modules / module_name
        assert base_path.is_file(), module_name
        assert shim_path.is_file(), module_name
        shim_text = shim_path.read_text(encoding="utf-8")
        assert "worldfoundry.base_models.diffusion_model.video.wan.wan_dreamzero.modules" in shim_text
        assert "class " not in shim_text
        assert "def " not in shim_text

    combined_base_text = "\n".join(
        (base_modules / module_name).read_text(encoding="utf-8") for module_name in moved_modules
    )
    assert "groot.vla.model.dreamzero.modules" not in combined_base_text

    action_head = (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/dreamzero/dreamzero_runtime/groot/vla/model/dreamzero/"
        "action_head/wan_flow_matching_action_tf.py"
    ).read_text(encoding="utf-8")
    assert "worldfoundry.base_models.diffusion_model.video.wan.wan_dreamzero.modules" in action_head
    assert "groot.vla.model.dreamzero.modules" not in action_head
