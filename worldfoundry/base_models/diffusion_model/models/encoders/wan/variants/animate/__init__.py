"""Wan Animate pose / motion conditioning helpers.

Subpackage entry for Animate recipes.  2-D pose
preprocessing lives in :mod:`.pose2d` (DWPose-style
keypoints to tensors).

Outputs feed Wan Animate control channels, not the UMT5
``context`` slot.  Text prompts still go through
:class:`~...component.WanTextConditioner`.

This is a control preprocessor family, not a text encoder.
"""

