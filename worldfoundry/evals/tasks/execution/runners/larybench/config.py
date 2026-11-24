import os
from dataclasses import dataclass, field
from pathlib import Path

from worldfoundry.core.io.paths import project_root as resolve_project_root


def _project_root() -> Path:
    return resolve_project_root(__file__)


def _config_root() -> Path:
    value = os.environ.get("LARY_CONFIG_ROOT")
    if value:
        return Path(value).expanduser()
    return _project_root() / "worldfoundry" / "data" / "benchmarks" / "assets" / "larybench" / "configs"


def _data_root() -> Path:
    return _project_root() / "worldfoundry" / "data" / "benchmarks" / "assets" / "larybench" / "metadata"


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass
class Config:
    project_root: Path = field(default_factory=_project_root)
    config_root: Path = field(default_factory=_config_root)
    log_dir: Path = field(default_factory=lambda: _path_from_env("LARY_LOG_DIR", "~/logs/LARY"))
    data_dir: Path = field(default_factory=lambda: _path_from_env("LARY_METADATA_DIR", _data_root()))

    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser()
        self.config_root = self.config_root.expanduser()
        self.log_dir = self.log_dir.expanduser()
        self.data_dir = self.data_dir.expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
