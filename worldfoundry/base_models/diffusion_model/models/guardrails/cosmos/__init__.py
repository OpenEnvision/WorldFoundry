"""NVIDIA Cosmos guardrail tables, SigLIP encoder, and face pixelation.

Category dicts feed the official prompt classifiers.  :class:`SigLIPEncoder`
embeds stills for vision-side safety.  :func:`pixelate_face` is the
post-decode face blur used when a frame is flagged.
"""
