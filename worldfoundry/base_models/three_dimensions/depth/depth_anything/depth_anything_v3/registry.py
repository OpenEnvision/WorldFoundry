# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from pathlib import Path

from worldfoundry.core.io.paths import resolve_data_path


def _candidate_config_dirs(package_file: str | Path) -> tuple[Path, ...]:
    package_configs = Path(package_file).resolve().parent / "configs"
    runtime_configs = resolve_data_path("models", "runtime", "configs", "depth_anything_v3", "configs")
    return (package_configs, runtime_configs)


def get_models_from_config_dir(configs_dir: str | Path) -> OrderedDict:
    configs_dir = Path(configs_dir)
    if not configs_dir.is_dir():
        return OrderedDict()
    model_entries = []
    for item in configs_dir.iterdir():
        if item.is_file() and item.suffix == ".yaml":
            model_entries.append((item.stem, str(item.resolve())))
    return OrderedDict(sorted(model_entries, key=lambda x: x[0]))


def get_models_for_package_file(package_file: str | Path) -> OrderedDict:
    for configs_dir in _candidate_config_dirs(package_file):
        model_entries = get_models_from_config_dir(configs_dir)
        if model_entries:
            return model_entries
    return OrderedDict()


def get_all_models() -> OrderedDict:
    """
    Scans all YAML files in the configs directory and returns a sorted dictionary where:
    - Keys are model names (YAML filenames without the .yaml extension)
    - Values are absolute paths to the corresponding YAML files
    """
    return get_models_for_package_file(__file__)


# Global registry for external imports
MODEL_REGISTRY = get_all_models()
