#!/usr/bin/env python3
"""Checks that metric scripts receive the automatically generated evaluation questions."""

from typing import Any


def ensure_auxiliary_info_compatibility(prompt_dict_ls: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    """Check that every prompt entry carries non-empty ``auxiliary_info`` questions.

    Args:
        prompt_dict_ls: Prompt dicts returned by ``load_dimension_info``.
        dimension: Evaluation dimension, used in messages.

    Returns:
        ``prompt_dict_ls`` unchanged.

    Raises:
        ValueError: An entry has no evaluation questions.
    """
    for i, prompt_dict in enumerate(prompt_dict_ls):
        if "auxiliary_info" not in prompt_dict or not prompt_dict["auxiliary_info"]:
            raise ValueError(
                f"Error: item {i + 1} of dimension '{dimension}' has no evaluation questions (auxiliary_info).\n"
                f"Generate the questions with automatic question generation first, or check the JSON file.\n"
                f"Hint: questions are generated automatically when the command runs; this error means the API call failed."
            )

    print(f"All {len(prompt_dict_ls)} items of dimension '{dimension}' have evaluation questions")
    return prompt_dict_ls


def get_default_questions_for_dimension(dimension: str) -> list[str]:
    """Return built-in evaluation questions for ``dimension`` (deprecated).

    Evaluation questions are generated through the API so that their quality is consistent; use automatic
    question generation instead.
    """
    default_questions = {
        "dynamic_attribute": [
            "Are there objects that change their appearance or properties throughout the video?",
            "Do any objects in the video transform, evolve, or modify their characteristics over time?",
            "Can you observe dynamic changes in the attributes of objects, such as color, size, shape, or texture?",
            "Are there elements in the video that demonstrate temporal variation in their visual properties?",
            "Do objects exhibit different states or configurations at different points in the video?",
        ],
        "camera_motion": [
            "Does the camera move during the video?",
            "Are there any zoom in or zoom out movements?",
            "Does the camera pan left or right?",
            "Are there any camera angle changes?",
            "Does the camera follow or track any objects?",
        ],
        "complex_plot": [
            "Does the video have a clear beginning?",
            "Are there characters performing actions?",
            "Does the story progress logically?",
            "Are there cause-and-effect relationships between events?",
            "Does the video have a resolution or conclusion?",
            "Do characters interact with objects or environment?",
            "Are there sequential events that build upon each other?",
            "Does the narrative maintain consistency?",
            "Are character motivations clear?",
            "Does the video convey a complete story or message?",
        ],
        "complex_landscape": [
            "Does the video show a complex natural environment?",
            "Are there multiple landscape elements visible?",
            "Does the scenery demonstrate depth and complexity?",
            "Are there varied terrain features in the video?",
            "Does the landscape appear realistic and detailed?",
        ],
        "motion_rationality": [
            "Do objects follow fundamental physical laws like gravity, inertia, and momentum conservation?",
            "Are movement trajectories smooth and continuous without sudden jumps or teleportation?",
            "Do collisions result in realistic reactions with proper force transfer and deformation?",
            "Are movement speeds reasonable and appropriate for the objects and environment shown?",
            "Does every motion have logical causes and predictable consequences?",
        ],
        "motion_order_understanding": [
            "Do events occur in a logical sequence?",
            "Are the motions and actions properly ordered?",
            "Does the temporal flow make sense?",
            "Are cause-and-effect relationships clear?",
            "Do actions follow a coherent progression?",
        ],
        "human_interaction": [
            "Are there people interacting in the video?",
            "Do human characters respond to each other?",
            "Are social behaviors realistic?",
            "Do interactions appear natural?",
            "Are emotional responses appropriate?",
        ],
        "multi_view_consistency": [
            "Are different viewpoints consistent?",
            "Do objects maintain their properties across views?",
            "Is the 3D geometry coherent?",
            "Are lighting and shadows consistent?",
            "Do perspective changes appear realistic?",
        ],
        "multi_view_semantic_consistency": [
            "Do objects maintain consistent 3D shape and structure across different viewpoints?",
            "Are relative positions and distances between objects preserved across different views?",
            "Do objects retain their identity, properties, and states consistently across viewpoints?",
            "Are lighting directions and shadow effects consistent with a coherent 3D scene?",
            "Do object visibility and occlusion relationships follow proper 3D spatial logic?",
        ],
        "composition": [
            "Is the visual composition well-balanced?",
            "Are elements arranged harmoniously?",
            "Does the framing enhance the content?",
            "Are visual elements properly proportioned?",
            "Does the composition guide viewer attention effectively?",
        ],
        "thermotics": [
            "Are thermal effects realistic?",
            "Do temperature-related phenomena appear accurate?",
            "Are heat transfer mechanisms properly depicted?",
            "Do thermal changes follow physical laws?",
            "Are temperature gradients visually consistent?",
        ],
        "physics_reasoning": [
            "Do objects follow fundamental physics principles like energy conservation and causality?",
            "Are spatial relationships and temporal sequences physically rational?",
            "Do physical events follow logical cause-and-effect sequences?",
            "Are force interactions and motion patterns realistic according to physics laws?",
            "Do light behavior, shadows, and optical phenomena appear physically accurate?",
            "Do material interactions and transformations follow real-world physics?",
            "Are thermal effects and phase transitions physically plausible?",
            "Does the overall scene maintain physical consistency and realism?",
            "Are object properties and behaviors consistent with their expected physics?",
            "Does the video demonstrate proper understanding of physical constraints?",
        ],
    }

    # Dimensions without dedicated questions get generic ones.
    return default_questions.get(
        dimension.lower(),
        [
            f"Does the video demonstrate good quality in terms of {dimension}?",
            f"Are the {dimension} aspects well-executed in the video?",
            f"Does the video meet expectations for {dimension}?",
            f"Are there clear examples of {dimension} in the video?",
            f"Is the {dimension} aspect consistent throughout the video?",
        ],
    )


def patch_metric_script_for_compatibility(dimension: str):
    """Hook for per-dimension compatibility patches, called when a metric script is imported; currently a no-op."""


# Built-in questions of the most common dimensions, for import by other scripts.
DEFAULT_QUESTIONS = {
    "dynamic_attribute": get_default_questions_for_dimension("dynamic_attribute"),
    "camera_motion": get_default_questions_for_dimension("camera_motion"),
    "complex_plot": get_default_questions_for_dimension("complex_plot"),
    "motion_rationality": get_default_questions_for_dimension("motion_rationality"),
}
