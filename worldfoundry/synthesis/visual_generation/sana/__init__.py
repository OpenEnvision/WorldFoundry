from worldfoundry.base_models.diffusion_model.recipes.sana_variants import (
    SANA_VARIANTS,
    SanaVariant,
    config_root,
    get_sana_variant,
    normalize_sana_model_id,
)

__all__ = [
    "SANA_VARIANTS",
    "SanaSynthesis",
    "SanaVariant",
    "config_root",
    "get_sana_variant",
    "normalize_sana_model_id",
]


def __getattr__(name):
    if name == "SanaSynthesis":
        from .sana_synthesis import SanaSynthesis

        return SanaSynthesis
    raise AttributeError(name)
