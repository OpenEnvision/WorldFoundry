from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .FaceScore import FaceScore as FaceScore
else:
    from FaceScore import FaceScore as FaceScore
