"""Thin re-export shim for the shared Inception I3D backbone.

Provenance: the pytorch-i3d network definition originally vendored here for
MiraBench now lives in ``worldfoundry.evaluation.tasks.metrics.fvd.pytorch_i3d``
so FVD/KVD share one backbone across benchmarks. This module only preserves the
official-protocol import path ``evaluation.pytorch_i3d``.
"""

from worldfoundry.evaluation.tasks.metrics.fvd.pytorch_i3d import (
    InceptionI3d,
    InceptionModule,
    MaxPool3dSamePadding,
    Unit3D,
)

__all__ = ["InceptionI3d", "InceptionModule", "MaxPool3dSamePadding", "Unit3D"]
