import os
import os.path as osp
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from downstream.downstream_datasets import ARDataset
from downstream.utils.query_utils import VIDEO_EXTS, encode_img_to_base64
from downstream.vlm import (
    CHOICES,
    COMMERCIAL_MODELS,
    LOCAL_MODELS,
    VIEW_ORDER,
    LMClassifier,
)

# from openeqa.utils.prompt_utils import load_prompt

UNIT_DISTANCE = 0.20
# unit_distance = 2.6
# unit_degree = 15.0
UNIT_DEGREE = 22.5
unit_degree_look = 15.0

# AR_ACTION_SPACE_ = [
#     f"go straight for {UNIT_DISTANCE:.2f}m",
#     f"turn left {UNIT_DEGREE:.1f} degrees and then go straight for {UNIT_DISTANCE:.2f}m",
#     f"turn left {UNIT_DEGREE * 2:.1f} degrees and then go straight for {UNIT_DISTANCE:.2f}m",
#     f"turn right {UNIT_DEGREE:.1f} degrees and then go straight for {UNIT_DISTANCE:.2f}m",
#     f"turn right {UNIT_DEGREE * 2:.1f} degrees and then go straight for {UNIT_DISTANCE:.2f}m",
# ]


def choice_text(choice_format, choice_texts):
    return "\n".join(
        f"{choice_mark}. {choice_text}"
        for choice_mark, choice_text in zip(
            CHOICES[choice_format],
            choice_texts,
            strict=False,
        )
    )


def construct_action_space_text(choice_format, include_stop=True):
    action_space = [
        f"go straight for {UNIT_DISTANCE:.2f}m",
        f"turn left {UNIT_DEGREE:.1f} degrees",
        f"turn right {UNIT_DEGREE:.1f} degrees",
    ]
    if include_stop:
        action_space.append("stop")
    if choice_format is not None:
        action_space = choice_text(choice_format, action_space)
    return action_space


AR_ACTION_SPACE = construct_action_space_text(choice_format=None, include_stop=True)
AR_ACTION_SPACE_NO_STOP = construct_action_space_text(choice_format=None, include_stop=False)

AR_ANSWER_SPACE = ARDataset.OBJECT_SET


CHOICE_EXAMPLES = {
    "digit": "['001','002','003','001']",
    "letter": "['A','B','C','A']",
}


def output_format_text(choice_format, output_type):
    if output_type == "answer":
        return f"### Output Format: \nOnly the <{choice_format}> to represent your choice."
    elif output_type == "N_action":
        return f"""### Output Format:\nReturn the sequence of the <{choice_format}> to represent the next low-level navigation action(s) to take, separated by `,` and bracketed by `[]`.

Example (next 4 predicted actions):
```
Question: ...
Answer: {CHOICE_EXAMPLES[choice_format]}
```
"""
    else:
        raise ValueError(f"Invalid output type: {output_type}")


surround_keys = [f"rgb_surround_{view}" for view in VIEW_ORDER]


def load_prompt_from(relative_path):
    prompt_folder = (Path(__file__).parent / "lm" / "prompts").resolve()
    return (prompt_folder / relative_path).read_text()


# fmt: off
TASK_PROMPTS = {
    "ar_answerer": load_prompt_from("task/ar_answerer.txt"),
    "ar_planner": load_prompt_from("task/ar_planner.txt"),
    "ignav_answerer": load_prompt_from("task/ignav_answerer.txt"),
    "ignav_planner": load_prompt_from("task/ignav_planner.txt"),
    "ignav_evaluator": load_prompt_from("task/ignav_evaluator.txt"),
    "aeqa_planner": load_prompt_from("task/aeqa_planner.txt"),
}

AUXILIARY_PROMPTS = {
    "ar_answerer": load_prompt_from("aux/ar_answerer.txt"),
    "ar_planner": load_prompt_from("aux/ar_planner.txt"),
    "ignav_evaluator_N_action": load_prompt_from("aux/ignav_evaluator_N_action.txt"),
    "aeqa_highlevel_planner": load_prompt_from("aux/aeqa_highlevel_planner.txt"),
    "vln_highlevel_planner": load_prompt_from("aux/aeqa_highlevel_planner.txt"),
    "objnav_highlevel_planner": load_prompt_from("aux/aeqa_highlevel_planner.txt"),
}

HIGH_LEVEL_PLANNER_PROMPT = {
    "aeqa_highlevel_planner": load_prompt_from("high_level/aeqa_highlevel_planner.txt"),
    "vln_highlevel_planner": load_prompt_from("high_level/vln_highlevel_planner.txt"),
    "objnav_highlevel_planner": load_prompt_from("high_level/objnav_highlevel_planner.txt"),
}
# fmt: on


def get_task_prompt(
    task_type,
    task_stage,
    output_space_name,
    output_space,
    output_format,
):
    return "\n\n".join(
        [
            TASK_PROMPTS[f"{task_type}_{task_stage}"],
            f"### {output_space_name}:\n{output_space}",
            output_format,
        ]
    )


def get_answerer_type_prompt(choice_format, task, choice_texts):
    assert choice_texts is not None
    task_name = task.split("_")[0]
    task_stage = task.split("_")[1]
    output_space = choice_text(choice_format, choice_texts)
    output_format = output_format_text(choice_format, "answer")

    return get_task_prompt(
        task_type=task_name,
        task_stage=task_stage,
        output_space_name="Answer space",
        output_space=output_space,
        output_format=output_format,
    )


def get_planner_N_type_prompt(choice_format, task, add_stop):
    task_name = task.split("_")[0]
    task_stage = task.split("_")[1]

    output_space = construct_action_space_text(choice_format, add_stop)
    output_format = output_format_text(choice_format, "N_action")

    return get_task_prompt(
        task_type=task_name,
        task_stage=task_stage,
        output_space_name="Action space",
        output_space=output_space,
        output_format=output_format,
    )


class PromptMixin:
    # ------------------------------------------------------------------ #
    # Message-building helpers (public because users may call them)
    # ------------------------------------------------------------------ #
    def _build_user_message(self, prompt: str, image_paths: str | Sequence[str]) -> dict[str, Any]:
        """
        Create a single 'user'-role OpenAI message with one prompt followed by N media items.
        Emits `image_url` for images and `video_url` for videos as expected by vLLM.
        """
        image_paths = [image_paths] if isinstance(image_paths, str) else image_paths
        self.__assert_paths_ok(image_paths)

        content = []
        if prompt:
            content.append({"type": "text", "text": str(prompt)})

        for p in image_paths:
            ext = os.path.splitext(p)[1].lower()
            media_content = {
                "url": self.__to_image_url(p),
                "detail": self.image_detail,
                "name": osp.relpath(p, start=os.getcwd()),
            }
            if ext in VIDEO_EXTS:
                # vLLM video input: {"type": "video_url", "video_url": {"url": ...}}
                content.append(
                    {
                        "type": "video_url",
                        "video_url": media_content,
                    }
                )
            else:
                # Image path (existing behavior)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": media_content,
                    }
                )

        return {"role": "user", "content": content}

    def _build_assistant_message(self, action: str) -> dict[str, str]:
        """Single-token assistant response (e.g. ‘A.’ or ‘stop.’)."""
        if len(action) != 1:
            raise ValueError(f"Expected single-character action, got: {action}")
        return {"role": "assistant", "content": f"{action[0]}."}

    # ------------------------------------------------------------------ #
    # Internal helpers (private)
    # ------------------------------------------------------------------ #
    def __assert_paths_ok(self, paths: Sequence[str]) -> None:
        """Fail fast if any path is missing or None."""
        for p in paths:
            if p is None:
                raise ValueError("Image path is required")
            if not osp.exists(p):
                raise FileNotFoundError(p)

    def __to_image_url(self, path: str) -> str:
        """Return either an absolute-file URL or a base64 data: URI (image or video)."""
        if self.image_use_abs:
            return f"file://{osp.abspath(path)}"
        return encode_img_to_base64(path)

    # ------------------------------------------------------------------ #
    # Message assembly (public)
    # ------------------------------------------------------------------ #
    def assemble_messages(
        self,
        state_traj: list[list[str]],
        action_traj: list[str],
        enable_history: bool,
        imagine_traj: list[str] = [],  # noqa: B006 - Read-only default; retain the public signature.
        imagine_action_traj: list[str] = [],  # noqa: B006 - Read-only default; retain the public signature.
        tobe_filled: dict = {},  # noqa: B006 - Read-only default; retain the public signature.
        enable_system_prompt: bool = False,
    ):
        """Build the *complete* message list in a readable, linear flow."""
        query_task = self.query_task
        if len(state_traj) != len(action_traj) + 1:
            raise ValueError("state_traj must be one longer than action_traj")

        messages = []
        # ---------- 1. optional system prompt ---------- #
        if enable_system_prompt:
            sys_prompt = self._get_task_prompt(query_task, tobe_filled)
            messages.append(self._build_user_message(sys_prompt, []))

        # ---------- 2. (optional) history ---------- #
        if enable_history:
            for state, action in zip(state_traj[:-1], action_traj, strict=False):
                if not enable_system_prompt:
                    task_prompt = self._get_task_prompt(query_task, tobe_filled)
                    messages.append(self._build_user_message(task_prompt, state))
                else:
                    messages.append(self._build_user_message("", state))

                messages.append(self._build_assistant_message(action))

        # ---------- 3. current observation ---------- #
        current_prompt = "" if enable_system_prompt else self._get_task_prompt(query_task, tobe_filled)
        messages.append(self._build_user_message(current_prompt, state_traj[-1]))

        # ---------- 4. imagined roll-outs (if any) ---------- #
        if imagine_traj and imagine_traj[-1]:
            aux_prompt = self._get_auxiliary_prompt(query_task)
            messages.append(self._build_user_message(aux_prompt, []))

            if imagine_action_traj:
                for plan, obs in zip(imagine_action_traj[-1], imagine_traj[-1], strict=False):
                    messages.append(self._build_user_message(plan, obs))
            else:
                for obs in imagine_traj[-1]:
                    messages.append(self._build_user_message("", obs))

        return messages


class VLM(LMClassifier, PromptMixin):
    # ---- division of different tasks ---- #
    _LOWLEVEL_PLANNER_TASKS: ClassVar[set[str]] = {
        "ar_planner_N_action",
        "aeqa_planner_N_action",
        "vln_planner_N_action",
        "objnav_planner_N_action",
        "ignav_planner_N_action",
        "ignav_evaluator_N_action",
    }
    _LOWLEVEL_PLANNER_TASKS_WITH_STOP: ClassVar[set[str]] = {
        "aeqa_planner_N_action",
        "vln_planner_N_action",
        "objnav_planner_N_action",
    }
    _LOWLEVEL_PLANNER_TASKS_NO_STOP: set[str] = _LOWLEVEL_PLANNER_TASKS - _LOWLEVEL_PLANNER_TASKS_WITH_STOP
    _HIGHLEVEL_PLANNER_TASKS: ClassVar[set[str]] = {
        "aeqa_highlevel_planner",
        "vln_highlevel_planner",
        "objnav_highlevel_planner",
    }
    _ANSWERER_TASKS: ClassVar[set[str]] = {
        "ar_answerer",
        "ar_planner",
        "ignav_answerer",
    }
    _OBSERVATION_KEY_PROMPT: ClassVar[dict[str, str]] = {
        # for AEQA:
        "['rgb_front']": "perspective view (Field of View (FOV) is 90 degrees)",
        # f"{surround_keys}": f"four perspective views (Field of View (FOV) is 90 degrees) in order of {VIEW_ORDER}",
        "['stitched_rgb']": f"four stitched perspective views (Field of View (FOV) is 90 degrees) in order of {VIEW_ORDER}",
        # for AR:
        "['rgb_bbox']": "equirectangular panorama",
        "['rgb_bbox_front']": "perspective view (Field of View (FOV) is 90 degrees)",
        "['rgb_bbox_front', 'rgb_bbox']": "perspective view (Field of View (FOV) is 90 degrees) and equirectangular panorama",
        "['rgb_bbox', 'rgb_bbox_front']": "perspective view (Field of View (FOV) is 90 degrees) and equirectangular panorama",
        # for IGNav:
        "['rgb_front', 'rgb']": "perspective view (Field of View (FOV) is 90 degrees) and equirectangular panorama",
        "['rgb', 'rgb_front']": "perspective view (Field of View (FOV) is 90 degrees) and equirectangular panorama",
    }

    def __init__(
        self,
        api_key,
        model,
        query_task,
        image_use_abs,
        classify_method,
        obs_key,
        categories=[],  # noqa: B006 - Read-only default; retain the public signature.
        top_logprobs=1,
        look_ahead_action_num=1,
        base_url=None,
    ):
        super().__init__(api_key, model, categories, top_logprobs, base_url=base_url)
        self.query_task = query_task
        self.image_use_abs = image_use_abs
        self.classify_method = classify_method
        self.obs_key = obs_key
        self.look_ahead_action_num = look_ahead_action_num

        if self.classify_method in ["classify_plain"]:
            assert self.model in LOCAL_MODELS
            assert len(categories) <= len(CHOICES["letter"])
            self.choice_format = "letter"
        elif self.classify_method == "classify":
            assert self.model in COMMERCIAL_MODELS
            assert len(categories) <= len(CHOICES["digit"])
            self.choice_format = "digit"
        elif self.classify_method is None:
            pass
        else:
            raise ValueError(f"Invalid classify method: {self.classify_method}")

    # ---------------------------------------------------------------------------------------- #
    # ----- FOR method that need to call accoding to "classify_method" and "query_task" -----
    # NOTE: This means if you want to add a new task, you should modify most of the code only below
    # ---------------------------------------------------------------------------------------- #
    def get_answer_key_for_return(self):
        if "aeqa" in self.query_task:
            answer_key = "Answer"
        elif any(x in self.query_task for x in ["vln", "objnav"]):
            answer_key = "Done"
        else:
            raise ValueError(f"Invalid task: {self.query_task}")
        return answer_key

    def _is_valide_actions_for_task(self, act_seq_len: int) -> bool:
        is_valid = False
        assert self.query_task in self._LOWLEVEL_PLANNER_TASKS, f"Invalid task: {self.query_task}"
        if "ignav" in self.query_task:
            # if act_seq_len == self.look_ahead_action_num: # for internvl3
            if act_seq_len <= self.look_ahead_action_num:  # for qwen2.5vl
                is_valid = True
        elif "aeqa" in self.query_task or "ar" in self.query_task:
            if act_seq_len <= self.look_ahead_action_num:
                is_valid = True
        else:
            raise ValueError(f"Invalid task: {self.query_task}")
        return is_valid

    # ------------ Query Dispatch (Main method 1) ------------ #
    def query_VLM(self, messages, tobe_filled={}, query_num=1):  # noqa: B006 - Read-only default; retain the public signature.
        if self.classify_method == "classify_plain":
            assert self.query_task in self._ANSWERER_TASKS
            response = self._classify_plain(messages, return_prob=True)

        elif self.classify_method is None:
            if self.query_task in self._LOWLEVEL_PLANNER_TASKS:
                response = self._query_next_N_action(messages, N=query_num)

            elif self.query_task in self._HIGHLEVEL_PLANNER_TASKS:
                response = self._query_next_instruction(
                    messages,
                    N=query_num,
                    input=tobe_filled,
                )

        return response

    # ------- Task-specific prompt generation (Main method 2) ------ #
    def _get_task_prompt(self, task: str, tobe_filled: dict[str, Any]) -> str:
        """
        A thin wrapper around the template functions / dictionaries that live
        elsewhere, plus variable substitution.
        """
        if self.classify_method == "classify_plain":
            assert task in self._ANSWERER_TASKS
            template = get_answerer_type_prompt(self.choice_format, task, self.categories)
        elif self.classify_method is None:
            if task in self._LOWLEVEL_PLANNER_TASKS:
                add_stop = task in self._LOWLEVEL_PLANNER_TASKS_WITH_STOP
                template = get_planner_N_type_prompt(self.choice_format, task, add_stop)
            elif task in self._HIGHLEVEL_PLANNER_TASKS:
                template = HIGH_LEVEL_PLANNER_PROMPT[task]
            else:
                raise KeyError(f"Unknown task: {task}")
        else:
            raise ValueError(f"Invalid classify method: {self.classify_method}")

        return template.format(
            look_ahead_action_num=self.look_ahead_action_num,
            obs_key=self._OBSERVATION_KEY_PROMPT[str(self.obs_key)],
            **tobe_filled,
        )

    def _get_auxiliary_prompt(self, task: str) -> str:
        """
        A thin wrapper around the auxiliary prompt templates that live
        elsewhere, plus variable substitution.
        """
        if task in AUXILIARY_PROMPTS:
            prompt = AUXILIARY_PROMPTS[task]
        else:
            raise KeyError(f"Unknown auxiliary task: {task}")
        return prompt
