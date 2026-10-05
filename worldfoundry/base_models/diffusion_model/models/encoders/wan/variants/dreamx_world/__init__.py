"""DreamX-World Wan text-conditioning components.

Subpackage entry for the DreamX UMT5 wrapper.
:class:`~.text_encoder.WanT5EncoderModel` is backed by the
canonical Wan T5 layers in :mod:`~...model`.

Prompt tokens still land in ``Conditioning.positive["context"]``
for DreamX world-model denoisers.  VAE side is
:mod:`~...autoencoders.wan.variants.dreamx_world`.

Compatibility naming only — not a new tokenizer.
"""

