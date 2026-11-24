"""
Case loader module.

Loads benchmark cases from JSON files. Each case defines:
- id, description, settings (scene, style, perspective, subject)
- interactions (multi-turn actions)
- eval_dimensions and eval_questions
"""

import glob
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_STYLES = {
    "realistic",
    "cartoon",
    "anime",
    "cinematic",
    "CG",
    "oil_painting",
    "ink",
    "pencil",
    "flat",
    "abstract",
}


class Interaction:
    def __init__(self, turn, type, action, prompt=None):
        self.turn = turn
        self.type = type
        self.action = action
        self.prompt = prompt


class Subject:
    def __init__(self, type, desc, movement):
        self.type = type
        self.desc = desc
        self.movement = movement


class Scene:
    def __init__(self, environment, attribute, name):
        self.environment = environment
        self.attribute = attribute
        self.name = name


class Rule:
    def __init__(self, type, desc):
        self.type = type
        self.desc = desc


class Settings:
    def __init__(self, scene, style, perspective, subject, rule, image_prompt, initial_image, subject_mask=None):
        self.scene = scene
        self.style = style
        self.perspective = perspective
        self.subject = subject
        self.rule = rule
        self.image_prompt = image_prompt
        self.initial_image = initial_image
        self.subject_mask = subject_mask


class EvalDimension:
    def __init__(self, enabled, focus):
        self.enabled = enabled
        self.focus = focus


class EvalQuestion:
    def __init__(self, dimension, question, expected, turn=None, sub_dim=None):
        self.dimension = dimension
        self.question = question
        self.expected = expected
        self.turn = turn
        self.sub_dim = sub_dim


class Case:
    def __init__(
        self,
        id,
        comment,
        description,
        settings,
        interactions,
        eval_dimensions,
        eval_questions,
        environment_prompt="",
        character_prompt="",
        perspective_prompt="",
    ):
        self.id = id
        self.comment = comment
        self.description = description
        self.environment_prompt = environment_prompt
        self.character_prompt = character_prompt
        self.perspective_prompt = perspective_prompt
        self.settings = settings
        self.interactions = interactions
        self.eval_dimensions = eval_dimensions
        self.eval_questions = eval_questions


def _migrate_case(case_dict: dict[str, Any]) -> Case:
    """Parse a case dict into Case object."""
    settings_dict = case_dict.get("settings", {})

    scene_dict = settings_dict.get("scene", {})
    scene = Scene(
        environment=scene_dict.get("environment", ""),
        attribute=scene_dict.get("attribute", ""),
        name=scene_dict.get("name", ""),
    )

    subject_dict = settings_dict.get("subject")
    subject = None
    if subject_dict:
        if "motion" in subject_dict:
            subject_dict["movement"] = subject_dict.pop("motion")
        subject = Subject(
            type=subject_dict.get("type", ""),
            desc=subject_dict.get("desc", ""),
            movement=subject_dict.get("movement", ""),
        )

    rule_dict = settings_dict.get("rule", {})
    rule = Rule(type=rule_dict.get("type", ""), desc=rule_dict.get("desc", ""))

    settings = Settings(
        scene=scene,
        style=settings_dict.get("style", ""),
        perspective=settings_dict.get("perspective", ""),
        subject=subject,
        rule=rule,
        image_prompt=settings_dict.get("image_prompt", ""),
        initial_image=settings_dict.get("initial_image"),
        subject_mask=settings_dict.get("subject_mask"),
    )

    interactions = []
    for inter_dict in case_dict.get("interactions", []):
        interactions.append(
            Interaction(
                turn=inter_dict.get("turn", 0),
                type=inter_dict.get("type", ""),
                action=inter_dict.get("action", ""),
                prompt=inter_dict.get("prompt", ""),
            )
        )

    eval_dims_dict = case_dict.get("eval_dimensions", {})

    def parse_eval_dim(key):
        value = eval_dims_dict.get(key, False)
        if isinstance(value, bool):
            return EvalDimension(enabled=value, focus=[])
        elif isinstance(value, dict):
            return EvalDimension(enabled=value.get("enabled", False), focus=value.get("focus", []))
        return EvalDimension(enabled=False, focus=[])

    eval_dimensions = {
        "video_quality": parse_eval_dim("video_quality"),
        "setting_adherence": parse_eval_dim("setting_adherence"),
        "interaction_adherence": parse_eval_dim("interaction_adherence"),
        "consistency": parse_eval_dim("consistency"),
        "causality": parse_eval_dim("causality"),
    }

    eval_questions = []
    for q_dict in case_dict.get("eval_questions", []):
        eval_questions.append(
            EvalQuestion(
                dimension=q_dict.get("dimension", ""),
                question=q_dict.get("question", ""),
                expected=q_dict.get("expected", ""),
                turn=q_dict.get("turn"),
                sub_dim=q_dict.get("sub_dim"),
            )
        )

    return Case(
        id=case_dict.get("id", 0),
        comment=case_dict.get("comment", ""),
        description=case_dict.get("description", ""),
        settings=settings,
        interactions=interactions,
        eval_dimensions=eval_dimensions,
        eval_questions=eval_questions,
        environment_prompt=case_dict.get("environment_prompt", ""),
        character_prompt=case_dict.get("character_prompt", ""),
        perspective_prompt=case_dict.get("perspective_prompt", ""),
    )


def _collect_json_files(path: str) -> list[str]:
    """Collect case JSON file paths from a file or directory."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        cases_dir = p / "cases"
        if cases_dir.is_dir():
            files = sorted(glob.glob(str(cases_dir / "case_*.json")))
        else:
            files = sorted(glob.glob(str(p / "case_*.json")))
        if not files:
            raise FileNotFoundError(f"No case_*.json files found in: {path}")
        return files
    raise FileNotFoundError(f"Path does not exist: {path}")


def load_cases(json_path: str) -> list[Case]:
    """Load cases from a JSON file or directory of case files."""
    files = _collect_json_files(json_path)
    cases = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            data = [data]
        for case_dict in data:
            cases.append(_migrate_case(case_dict))
    logger.info(f"Loaded {len(cases)} cases from: {json_path}")
    return cases


def load_cases_raw(json_path: str) -> list[dict]:
    """Load cases as raw dicts."""
    files = _collect_json_files(json_path)
    cases = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, list):
            cases.extend(data)
        elif isinstance(data, dict):
            cases.append(data)
    return cases
