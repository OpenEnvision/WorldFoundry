"""Shared image, audio and video infrastructure.

codecs owns encoding, decoding and frame sampling; processing owns tiling,
display conversion and streaming post-processing. Media types, resolutions and
artifact previews live at this package root. Model-specific preprocessing stays
with its model family. Importing this package loads no optional media stacks.
"""
