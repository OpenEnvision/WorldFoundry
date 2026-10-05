# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import html
import string

import ftfy
import regex as re
from transformers import AutoTokenizer

__all__ = ['HuggingfaceTokenizer']


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import HuggingfaceTokenizer
