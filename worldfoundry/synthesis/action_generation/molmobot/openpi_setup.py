"""Install the pinned OpenPI Transformers overrides in a dedicated environment."""

from __future__ import annotations

import shutil
from pathlib import Path


def install() -> list[str]:
    import openpi.models_pytorch.transformers_replace as replacements
    import transformers

    if transformers.__version__ != "4.53.2":
        raise RuntimeError("MolmoBot-Pi0 requires Transformers 4.53.2 before applying OpenPI overrides.")
    source = Path(replacements.__path__[0])
    target_root = Path(transformers.__path__[0])
    changed: list[str] = []
    for replacement in sorted(source.rglob("*.py")):
        relative = replacement.relative_to(source)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Package managers may symlink or hardlink individual files to a
        # shared cache.  Replace the directory entry before copying so an
        # override cannot modify another environment through either link.
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(replacement, target)
        changed.append(str(relative))
    return changed


def main() -> None:
    changed = install()
    print(f"Installed {len(changed)} OpenPI Transformers overrides: {', '.join(changed)}")


if __name__ == "__main__":
    main()
