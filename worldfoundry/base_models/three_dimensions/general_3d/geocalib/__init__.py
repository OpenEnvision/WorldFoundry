# This file includes code originally from the Geocalib repository:
# https://github.com/cvg/GeoCalib
# Licensed under the Apache-2.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""GeoCalib and AutoLevel base-model integrations."""

__all__ = ["FlowEstimator", "GeoCalib", "preprocess_image"]


def __getattr__(name):
    if name == "GeoCalib":
        from .extractor import GeoCalib

        return GeoCalib
    if name in {"FlowEstimator", "preprocess_image"}:
        from .autolevel import FlowEstimator, preprocess_image

        return {"FlowEstimator": FlowEstimator, "preprocess_image": preprocess_image}[name]
    raise AttributeError(name)
