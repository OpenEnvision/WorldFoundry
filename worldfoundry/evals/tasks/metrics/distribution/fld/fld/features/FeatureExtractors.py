from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .CLIPFeatureExtractor import CLIPFeatureExtractor
    from .DINOv2FeatureExtractor import DINOv2FeatureExtractor
    from .InceptionFeatureExtractor import InceptionFeatureExtractor
else:
    from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
    from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
    from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor

ALL_FEATURE_EXTRACTORS = [
    InceptionFeatureExtractor(),
    CLIPFeatureExtractor(),
    DINOv2FeatureExtractor(),
]

MAIN_FEATURE_EXTRACTORS = [
    InceptionFeatureExtractor(),
    DINOv2FeatureExtractor(),
]
