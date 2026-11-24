"""Object detection success-rate helpers adapted from WorldScore."""

from __future__ import annotations

import re
from collections.abc import Iterable


def standardize_string(value: str) -> str:
    return re.sub(r"\s*([_\-*/+])\s*", r"\1", value)


def extract_noun_adjective(phrase: str) -> tuple[list[str], list[str]]:
    try:
        spacy = __import__("spacy")
    except ImportError as exc:
        raise ImportError("object_detection helpers require spacy (en_core_web_sm).") from exc

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(phrase)
    adjectives: list[str] = []
    nouns: list[str] = []
    for token in doc:
        if token.pos_ in ["ADJ", "JJ", "JJR", "JJS"]:
            adjectives.append(token.text.lower())
        elif token.pos_ in ["NOUN", "PROPN", "NN", "NNS", "NNP", "NNPS"]:
            nouns.append(token.text.lower())
        elif token.pos_ == "VERB" and token.tag_ == "VBN":
            adjectives.append(token.text.lower())
    return adjectives, nouns


def get_match_point(result: str, prompt_list: Iterable[str]) -> tuple[int, set[str]]:
    matched_prompts: set[str] = set()
    prompts = list(prompt_list)
    if "##" in result:
        result = result.replace("##", "")
        for prompt in prompts:
            if result in prompt:
                matched_prompts.add(prompt)
                return 1, matched_prompts
    else:
        result_adjectives, result_nouns = extract_noun_adjective(result)
        for prompt in prompts:
            prompt_adjectives, prompt_nouns = extract_noun_adjective(prompt)
            noun_matches = sum(1 for noun in result_nouns if noun in prompt_nouns)
            if noun_matches == 0:
                continue
            matched_prompts.add(prompt)
            return 1, matched_prompts
    return 0, matched_prompts


def compute_object_detection_success_rate(
    detected_phrases: list[str],
    prompt_list: list[str],
) -> float:
    """Compute WorldScore-style object detection success rate in [0, 1]."""
    prompts = [p.lower() for p in prompt_list]
    if not prompts:
        return 1.0
    count = 0
    result_cache: set[str] = set()
    remaining = set(prompts)
    for phrase in detected_phrases:
        result = re.sub(r"\(.*?\)", "", phrase).strip().lower()
        result = standardize_string(result)
        if result in result_cache or result not in remaining:
            continue
        result_cache.add(result)
        if result in remaining:
            count += 1
            remaining.remove(result)
    for phrase in detected_phrases:
        result = re.sub(r"\(.*?\)", "", phrase).strip().lower()
        result = standardize_string(result)
        if result not in result_cache:
            point, matched = get_match_point(result, list(remaining))
            count += point
            remaining -= matched
    return float(min(count / len(prompts), 1.0))
