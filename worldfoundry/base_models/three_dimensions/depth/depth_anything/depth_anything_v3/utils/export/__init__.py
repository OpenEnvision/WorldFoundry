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

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.specs import Prediction


def _import_exporter(export_format: str):
    try:
        if export_format == "glb":
            from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.export_glb import (
                export_to_glb,
            )

            return export_to_glb
        if export_format == "mini_npz":
            from .npz import export_to_mini_npz

            return export_to_mini_npz
        if export_format == "npz":
            from .npz import export_to_npz

            return export_to_npz
        if export_format == "feat_vis":
            from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.visualize import (
                export_to_feat_vis,
            )

            return export_to_feat_vis
        if export_format == "depth_vis":
            from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.visualize import (
                export_to_depth_vis,
            )

            return export_to_depth_vis
        if export_format == "gs_ply":
            from .gs import export_to_gs_ply

            return export_to_gs_ply
        if export_format == "gs_video":
            from .gs import export_to_gs_video

            return export_to_gs_video
        if export_format == "colmap":
            from .colmap import export_to_colmap

            return export_to_colmap
    except ModuleNotFoundError as exc:
        missing = exc.name or "optional dependency"
        raise ImportError(
            f"DepthAnything3 export format '{export_format}' requires optional dependency "
            f"'{missing}'. Install the upstream DA3 export extras before using this format."
        ) from exc

    raise ValueError(f"Unsupported export format: {export_format}")


def export(
    prediction: Prediction,
    export_format: str,
    export_dir: str,
    **kwargs,
):
    if "-" in export_format:
        export_formats = export_format.split("-")
        for export_format in export_formats:
            export(prediction, export_format, export_dir, **kwargs)
        return  # Prevent falling through to single-format handling

    exporter = _import_exporter(export_format)
    if export_format in {"mini_npz", "npz", "depth_vis"}:
        exporter(prediction, export_dir)
    else:
        exporter(prediction, export_dir, **kwargs.get(export_format, {}))


__all__ = [
    export,
]
