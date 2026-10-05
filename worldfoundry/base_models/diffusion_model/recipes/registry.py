"""Instance-local registry of declarative native recipes.

Sits between spec and assembler: lazy-import the model stack, then hand a recipe
to the assembler to produce a runner.

    default_native_diffusion_registry() freezes a lazy table
        → resolve(model_id) yields NativeDiffusionRecipe
        → build_runner(...) calls NativeDiffusionAssembler.build
        → strategy / runner

Inputs: identities and providers from register / register_lazy; model_id for
resolve / build_runner.
Outputs: NativeDiffusionRecipe or DiffusionExecutor.
Must not mutate a process-wide singleton, force-import optional model stacks at
import time, or let a model package register a strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib import import_module

from ..assembly import NativeDiffusionAssembler
from ..components import ComponentKey
from ..extensions import DiffusionExtension
from ..loaders import CheckpointSpec
from ..optimizations import RuntimePolicy
from ..runners import DiffusionExecutor
from .spec import NativeDiffusionRecipe


def _key(value: str) -> str:
    """Normalize model_id / alias so ``Wan2.1`` and ``wan2.1`` collide on one key."""
    return str(value).strip().lower().replace("_", "-")


class DuplicateNativeDiffusionRecipeError(ValueError):
    """Raised when recipe identifiers or aliases collide, or one recipe repeats its own keys."""


class UnknownNativeDiffusionRecipeError(KeyError):
    """Raised when no native recipe owns the requested identifier (including aliases)."""


class NativeDiffusionRegistry:
    """Instance-local recipe table backed by one NativeDiffusionAssembler.

    Internal maps:
        _recipes: already-materialized NativeDiffusionRecipe values;
        _providers: lazy providers; the model stack is imported only on resolve;
        _model_ids: canonical key → original model_id string (for model_ids());
        _aliases: any normalized key → canonical model_key;
        _frozen: after freeze(), further register calls are rejected.

    This class only resolves identity. Component / runner construction is always
    delegated to the assembler.
    """

    def __init__(self, assembler: NativeDiffusionAssembler | None = None) -> None:
        self._assembler = assembler or NativeDiffusionAssembler()
        self._recipes: dict[str, NativeDiffusionRecipe] = {}
        self._providers: dict[str, Callable[[], NativeDiffusionRecipe]] = {}
        self._model_ids: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._frozen = False

    def register(self, recipe: NativeDiffusionRecipe) -> None:
        """Register an already-built recipe. Frozen → :exc:`RuntimeError`; collisions → :exc:`DuplicateNativeDiffusionRecipeError`."""
        model_key = _key(recipe.model_id)
        self._reserve(model_key, recipe.model_id, recipe.aliases)
        self._recipes[model_key] = recipe

    def register_lazy(
        self,
        model_id: str,
        provider: Callable[[], NativeDiffusionRecipe],
        *,
        aliases: Iterable[str] = (),
    ) -> None:
        """Reserve identity and aliases without importing an optional model stack (Wan, Hunyuan, …).

        The real NativeDiffusionRecipe is produced only when :meth:`resolve` calls the provider.
        A non-callable provider raises :exc:`TypeError`.
        """

        if not callable(provider):
            raise TypeError("native diffusion recipe provider must be callable")
        model_key = _key(model_id)
        self._reserve(model_key, model_id, tuple(aliases))
        self._providers[model_key] = provider

    def _reserve(self, model_key: str, model_id: str, aliases: Iterable[str]) -> None:
        """Reserve model_id and aliases in _aliases for later lazy identity checks."""
        if self._frozen:
            raise RuntimeError("native diffusion registry is frozen")
        keys = (model_key, *(_key(alias) for alias in aliases))
        if any(not key for key in keys):
            raise ValueError("model aliases cannot be empty")
        if len(keys) != len(set(keys)):
            raise DuplicateNativeDiffusionRecipeError(f"native diffusion model contains duplicate keys: {keys}")
        collisions = [key for key in keys if key in self._aliases]
        if collisions:
            raise DuplicateNativeDiffusionRecipeError(f"native diffusion recipe keys already registered: {collisions}")
        self._model_ids[model_key] = str(model_id)
        for alias in keys:
            self._aliases[alias] = model_key

    def freeze(self) -> None:
        """Reject further register / register_lazy so import side effects cannot sneak in models."""
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def resolve(self, model_id: str) -> NativeDiffusionRecipe:
        """Return the recipe for an ID or alias. Lazy entries run the provider and identity checks here."""
        key = _key(model_id)
        try:
            canonical = self._aliases[key]
        except KeyError as error:
            raise UnknownNativeDiffusionRecipeError(model_id) from error
        recipe = self._recipes.get(canonical)
        if recipe is not None:
            return recipe
        try:
            provider = self._providers[canonical]
        except KeyError as error:
            raise RuntimeError(f"native diffusion recipe {canonical!r} has no provider") from error
        recipe = provider()
        if not isinstance(recipe, NativeDiffusionRecipe):
            raise TypeError(
                f"provider for {canonical!r} returned {type(recipe).__name__}, expected NativeDiffusionRecipe"
            )
        # Lazy identity check: the provider's model_id + aliases must match the
        # set reserved by register_lazy exactly, so a delayed import cannot rename
        # the recipe and leave aliases pointing at the wrong model.
        actual_keys = {_key(recipe.model_id), *(_key(alias) for alias in recipe.aliases)}
        reserved_keys = {alias for alias, owner in self._aliases.items() if owner == canonical}
        if _key(recipe.model_id) != canonical or actual_keys != reserved_keys:
            raise ValueError(
                f"lazy recipe identity mismatch for {canonical!r}: "
                f"reserved={sorted(reserved_keys)}, actual={sorted(actual_keys)}"
            )
        self._recipes[canonical] = recipe
        return recipe

    def build_runner(
        self,
        model_id: str,
        *,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        extensions: Iterable[DiffusionExtension] = (),
    ) -> DiffusionExecutor:
        """Resolve, then ask the assembler for the full component + strategy → runner inference path."""
        return self._assembler.build(
            self.resolve(model_id),
            policy=policy,
            checkpoint_overrides=checkpoint_overrides,
            component_options=component_options,
            extensions=extensions,
        )

    def model_ids(self) -> tuple[str, ...]:
        """Reserved original model_id values (including unresolved lazy entries), sorted."""
        return tuple(sorted(self._model_ids.values()))


def _lazy_recipe(module_name: str, factory_name: str) -> Callable[[], NativeDiffusionRecipe]:
    """Import ``module_name`` relative to this package, then call ``factory_name()``."""
    def load() -> NativeDiffusionRecipe:
        module = import_module(module_name, __package__)
        return getattr(module, factory_name)()

    return load


def _lazy_echo_recipe(model_id: str) -> Callable[[], NativeDiffusionRecipe]:
    """Echo-Memory variant: fetch the denoiser spec, then call echo_memory_recipe."""
    def load() -> NativeDiffusionRecipe:
        module = import_module(".echo_memory", __package__)
        specs = import_module("..models.denoisers.echo_memory_spec", __package__)
        return module.echo_memory_recipe(specs.get_echo_memory_model_spec(model_id))

    return load


def _lazy_sana_recipe(model_id: str) -> Callable[[], NativeDiffusionRecipe]:
    """Table-driven lazy SANA provider so the full weight stack is not imported before freeze."""
    def load() -> NativeDiffusionRecipe:
        module = import_module(".sana", __package__)
        return module.sana_recipe(model_id)

    return load


def default_native_diffusion_registry() -> NativeDiffusionRegistry:
    """Build and freeze the inference-ready built-in recipe table. Each call is a new instance."""

    registry = NativeDiffusionRegistry()

    entries = (
        (
            "wan2.1-vace",
            ".wan",
            "wan21_vace_14b_recipe",
            ("wan-vace", "wan2.1-vace-14b", "Wan-AI/Wan2.1-VACE-14B"),
        ),
        (
            "wan2.1-t2v-1.3b",
            ".wan",
            "wan21_t2v_1p3b_recipe",
            (
                "wan2.1",
                "wan-2.1",
                "wan2p1",
                "wan2.1-t2v",
                "wan2p1-t2v",
                "wan21-t2v-1.3b",
                "Wan-AI/Wan2.1-T2V-1.3B",
            ),
        ),
        (
            "wan2.1-t2v-14b",
            ".wan",
            "wan21_t2v_14b_recipe",
            ("wan21-t2v-14b", "Wan-AI/Wan2.1-T2V-14B"),
        ),
        (
            "wan2.1-i2v-14b-480p",
            ".wan",
            "wan21_i2v_14b_480p_recipe",
            (
                "wan2.1-i2v",
                "wan2p1-i2v",
                "wan21-i2v-480p",
                "Wan-AI/Wan2.1-I2V-14B-480P",
            ),
        ),
        (
            "wan2.1-i2v-14b-720p",
            ".wan",
            "wan21_i2v_14b_720p_recipe",
            ("wan21-i2v-720p", "Wan-AI/Wan2.1-I2V-14B-720P"),
        ),
        (
            "wan2.2-t2v-a14b",
            ".wan",
            "wan22_t2v_a14b_recipe",
            ("Wan-AI/Wan2.2-T2V-A14B",),
        ),
        (
            "wan2.2-i2v-a14b",
            ".wan",
            "wan22_i2v_a14b_recipe",
            ("Wan-AI/Wan2.2-I2V-A14B",),
        ),
        (
            "wan2.2-ti2v-5b",
            ".wan",
            "wan22_ti2v_5b_recipe",
            (
                "wan2.2",
                "wan-2.2",
                "wan2p2",
                "wan2.2-ti2v-5b-1280x704-121f",
                "Wan-AI/Wan2.2-TI2V-5B",
            ),
        ),
        (
            "fastvideo-causal-wan2.2-i2v-14b",
            ".wan",
            "fastvideo_causal_wan22_i2v_14b_recipe",
            ("FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers",),
        ),
        (
            "t2v_turbo_t2v",
            ".t2v_turbo",
            "t2v_turbo_t2v_recipe",
            ("t2v-turbo", "VideoCrafter/VideoCrafter2", "jiachenli-ucsb/T2V-Turbo-VC2"),
        ),
        (
            "vchitect-2-t2v",
            ".vchitect",
            "vchitect_2_t2v_recipe",
            ("vchitect", "vchitect-2", "Vchitect/Vchitect-2.0-2B"),
        ),
        (
            "step-video-t2v",
            ".step_video",
            "step_video_t2v_recipe",
            ("stepvideo", "step-video", "stepvideo-t2v", "stepfun-ai/stepvideo-t2v"),
        ),
        (
            "skyreels-v2",
            ".skyreels",
            "skyreels_v2_recipe",
            ("skyreels2", "skyreels-v2-t2v", "Skywork/SkyReels-V2-DF-1.3B-540P"),
        ),
        (
            "skyreels-v3",
            ".skyreels",
            "skyreels_v3_recipe",
            (
                "skyreels-v3-r2v",
                "skyreels-v3-reference-to-video",
                "Skywork/SkyReels-V3-R2V-14B",
            ),
        ),
        (
            "matrix-game-3.5-first-person",
            ".matrix_game",
            "matrix_game_35_first_person_recipe",
            ("RiemannDynamics/Matrix-Game-3.5-Base:first-person",),
        ),
        (
            "matrix-game-3.5-third-person",
            ".matrix_game",
            "matrix_game_35_third_person_recipe",
            ("RiemannDynamics/Matrix-Game-3.5-Base:third-person",),
        ),
        ("ltx-2-i2v", ".ltx", "ltx2_i2v_recipe", ("ltx2-i2v",)),
        ("ltx-2-t2v", ".ltx", "ltx2_t2v_recipe", ("ltx2-t2v",)),
        ("ltx-2.3-i2v", ".ltx", "ltx23_i2v_recipe", ("ltx2.3-i2v", "ltx2_3_i2v")),
        ("ltx-2.3-t2v", ".ltx", "ltx23_t2v_recipe", ("ltx2.3-t2v", "ltx2_3_t2v")),
        ("ltx-video-i2v", ".ltx", "ltx_video_i2v_recipe", ("ltx-video",)),
        (
            "hello-world",
            ".hello_world",
            "hello_world_recipe",
            ("helloworld", "AlayaLab/HelloWorld"),
        ),
        (
            "hunyuanvideo-t2v",
            ".hunyuan_video",
            "hunyuan_video_t2v_recipe",
            ("hunyuanvideo", "tencent/HunyuanVideo"),
        ),
        (
            "hunyuanvideo-i2v",
            ".hunyuan_video",
            "hunyuan_video_i2v_recipe",
            ("tencent/HunyuanVideo-I2V",),
        ),
        (
            "hunyuanvideo-1.5-t2v",
            ".hunyuan_video",
            "hunyuan_video15_t2v_recipe",
            ("hunyuanvideo-1.5", "tencent/HunyuanVideo-1.5"),
        ),
        ("hunyuanvideo-1.5-i2v", ".hunyuan_video", "hunyuan_video15_i2v_recipe", ()),
        (
            "gen3c-cosmos1-7b",
            ".cosmos1",
            "gen3c_recipe",
            ("gen3c", "cosmos1-gen3c", "cosmos-predict1-gen3c", "nvidia/GEN3C-Cosmos-7B"),
        ),
        (
            "cosmos-predict2-2b-video2world",
            ".cosmos2",
            "cosmos2_2b_video2world_recipe",
            (
                "cosmos-predict2",
                "cosmos-predict-2",
                "cosmos2",
                "nvidia/Cosmos-Predict2-2B-Video2World",
            ),
        ),
        (
            "cosmos-predict2-14b-video2world",
            ".cosmos2",
            "cosmos2_14b_video2world_recipe",
            ("cosmos-predict2-14b", "nvidia/Cosmos-Predict2-14B-Video2World"),
        ),
        ("cosmos3-nano", ".cosmos3", "cosmos3_nano_recipe", ("cosmos3",)),
        ("cosmos3-super", ".cosmos3", "cosmos3_super_recipe", ()),
        (
            "cosmos-predict2.5-2b",
            ".cosmos2p5",
            "cosmos25_2b_recipe",
            ("cosmos-predict2.5", "cosmos-predict2p5", "nvidia/Cosmos-Predict2.5-2B"),
        ),
        (
            "cosmos-predict2.5-14b",
            ".cosmos2p5",
            "cosmos25_14b_recipe",
            ("nvidia/Cosmos-Predict2.5-14B",),
        ),
        (
            "cosmos-transfer2.5-2b-controlled-video",
            ".cosmos2p5",
            "cosmos25_transfer_2b_recipe",
            (
                "cosmos-transfer-2.5",
                "cosmos-transfer2.5",
                "cosmos-transfer2p5",
                "cosmos-transfer-2.5-2b",
                "nvidia/Cosmos-Transfer2.5-2B",
            ),
        ),
        (
            "gamma-world-causal-few-step",
            ".gamma_world",
            "gamma_world_causal_few_step_recipe",
            ("gamma-world", "gammaworld"),
        ),
        (
            "gamma-world-causal",
            ".gamma_world",
            "gamma_world_causal_recipe",
            (),
        ),
        (
            "gamma-world-bidirectional",
            ".gamma_world",
            "gamma_world_bidirectional_recipe",
            (),
        ),
        (
            "sana-wm",
            ".sana",
            "sana_world_recipe",
            ("sana-wm-2.6b", "Efficient-Large-Model/SANA-WM_bidirectional"),
        ),
        (
            "sana-wm-streaming",
            ".sana",
            "sana_wm_streaming_recipe",
            ("Efficient-Large-Model/SANA-WM_streaming",),
        ),
    )
    for model_id, module_name, factory_name, aliases in entries:
        registry.register_lazy(
            model_id,
            _lazy_recipe(module_name, factory_name),
            aliases=aliases,
        )

    sana_variants = import_module(".sana_variants", __package__)
    for model_id in sana_variants.SANA_VARIANTS:
        aliases = tuple(
            alias
            for alias, target in sana_variants.SANA_ALIASES.items()
            if target == model_id
        )
        registry.register_lazy(model_id, _lazy_sana_recipe(model_id), aliases=aliases)

    echo_entries = (
        ("echo-memory-context-k1", ("echo-context-k1",)),
    )
    for model_id, aliases in echo_entries:
        registry.register_lazy(model_id, _lazy_echo_recipe(model_id), aliases=aliases)
    registry.freeze()
    return registry


__all__ = [
    "DuplicateNativeDiffusionRecipeError",
    "NativeDiffusionRegistry",
    "UnknownNativeDiffusionRecipeError",
    "default_native_diffusion_registry",
]
