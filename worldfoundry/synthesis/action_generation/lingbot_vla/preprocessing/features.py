"""LingBot-VLA v1 observation preprocessing and action postprocessing."""

import json
from collections import OrderedDict

import torch
import torch.nn.functional as F
import yaml

from worldfoundry.base_models.llm_mllm_core.action.lingbot.preprocessing import (
    FeatureInfo,
    prepare_action,
    prepare_state,
)

from .transforms import Normalizer, prepare_images, prepare_joint_pad, prepare_language


class FeatureTransform:
    def __init__(
        self,
        robot_config_path,
        data_config,
        tokenizer,
        image_processor,
        do_nomalize=True,
        chunk_size=50,
        use_depth_align=False,
        norm_stats_path=None,
        load_image=True,
    ):
        with open(robot_config_path) as f:
            robot_config = yaml.safe_load(f)
        f.close()

        # Initialize feature config from data_config (joints and cameras) if available.
        self.feature_config = FeatureInfo()
        if getattr(data_config, "joints", None) is not None:
            self.feature_config.update_info(data_config)

        self.data_config = data_config

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.load_image = load_image
        if not self.load_image:
            assert not do_nomalize

        self.chunk_size = chunk_size
        self.return_item_befor_padding = not do_nomalize
        self.use_depth_align = use_depth_align

        self.check_robot_config(robot_config)
        # keep the self.feature_to_keep in lerobot item when convert to new item
        self.feature_to_keep = {"timestamp", "frame_index", "episode_index", "task_index", "task", "action_is_pad"}

        target_features = {"states": [], "actions": [], "images": []}
        # Lists preserve robot-config camera/state ordering. A set here makes
        # list-valued observations bind to different cameras across processes.
        org_features = {"states": [], "actions": [], "images": []}
        self.get_feature_mapping(robot_config, target_features, org_features)
        self.states = target_features["states"]
        self.actions = target_features["actions"]
        self.images = target_features["images"]

        self.org_features = org_features
        self.normalizer = self.get_normalizer(norm_stats_path, do_nomalize, data_config)

    def get_normalizer(self, norm_stats_path, do_nomalize, data_config):
        # Build a Normalizer from pre-computed statistics (mean/std/bounds) for actions and states.
        # Image features use identity normalization (no-op).
        if not do_nomalize:
            return None

        print(f"Loading normalization stats from: {norm_stats_path}")
        action_state_norm_type = data_config.norm_type
        assert norm_stats_path is not None
        norm_type = {}
        for feature in self.actions + self.states:
            norm_type[feature] = action_state_norm_type
        for feature in self.images:
            norm_type[feature] = "identity"

        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        f.close()

        normalizer = Normalizer(
            norm_stats=norm_stats["norm_stats"],
            norm_type=norm_type,
        )
        return normalizer

    def check_robot_config(self, robot_config):
        # Validate that all action, state, and image features listed in the robot config
        # are included in the predefined feature set from the training data config.
        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)
            for feature_convert_info in features_convert_info:
                if isinstance(feature_convert_info, dict):
                    assert len(feature_convert_info.keys()) == 1

                if feature_category == "actions":
                    assert isinstance(feature_convert_info, dict)
                    action_feature = list(feature_convert_info.keys())[0].split("action.")[-1]  # noqa: RUF015 - Keeps eager list construction and IndexError.
                    if action_feature not in self.feature_config.joints:
                        raise ValueError(
                            f"{action_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}"
                        )

                elif feature_category == "states":
                    assert isinstance(feature_convert_info, (dict, str))
                    if isinstance(feature_convert_info, dict):
                        state_feature = list(feature_convert_info.keys())[0]  # noqa: RUF015 - Keeps eager list construction and IndexError.
                    else:
                        state_feature = feature_convert_info
                    state_feature = state_feature.split("observation.state.")[-1]

                    if state_feature not in self.feature_config.joints:
                        raise ValueError(
                            f"{state_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}"
                        )

                elif feature_category == "images":
                    assert isinstance(feature_convert_info, (dict, str))
                    if isinstance(feature_convert_info, dict):
                        image_feature = list(feature_convert_info.keys())[0]  # noqa: RUF015 - Keeps eager list construction and IndexError.
                    else:
                        image_feature = feature_convert_info
                    if image_feature not in self.feature_config.images:
                        raise ValueError(
                            f"{image_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.images}"
                        )

    def get_feature_mapping(self, robot_config, target_features, org_features):
        # Build bidirectional mappings between original dataset feature keys and target feature keys.
        # Handles feature renaming, slicing, and concatenation as defined in the robot config.
        # Also records which action features should subtract corresponding state values.
        to_convert_features = {}
        reverse_convert_features = {}
        action_subtract_state = {}
        actions_convert_from_state = set()

        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)

            for feature_convert_info in features_convert_info:
                if isinstance(feature_convert_info, str):
                    self.feature_to_keep.add(feature_convert_info)
                    target_features[feature_category].append(feature_convert_info)
                    if feature_convert_info not in org_features[feature_category]:
                        org_features[feature_category].append(feature_convert_info)

                elif isinstance(feature_convert_info, dict):
                    target_feature = next(iter(feature_convert_info.keys()))
                    target_features[feature_category].append(target_feature)

                    if feature_category == "actions":
                        assert isinstance(feature_convert_info, dict)
                        # Set subtract_state to False by default.
                        action_subtract_state[target_feature] = feature_convert_info[target_feature].pop(
                            "subtract_state", False
                        )
                        if "end.position" in target_feature or "effector.position" in target_feature:
                            assert not action_subtract_state[target_feature]

                        if_convert_from_state = feature_convert_info[target_feature].pop("convert_from_state", False)
                        if if_convert_from_state:
                            actions_convert_from_state.add(target_feature)

                        if "origin_keys" not in feature_convert_info[target_feature] and not if_convert_from_state:
                            self.feature_to_keep.add(target_feature)

                    if "origin_keys" in feature_convert_info[target_feature]:
                        if isinstance(feature_convert_info[target_feature]["origin_keys"], list):
                            ordered_origin_keys = OrderedDict()
                            for item in feature_convert_info[target_feature]["origin_keys"]:
                                for k, v in item.items():
                                    while k in ordered_origin_keys:
                                        k = k + "*"
                                    ordered_origin_keys[k] = v

                            feature_convert_info[target_feature]["origin_keys"] = ordered_origin_keys
                            to_convert_features.update(feature_convert_info)

                            if feature_category in ["actions", "states"]:
                                for key in ordered_origin_keys:
                                    key = key.split("*")[0]
                                    if key not in org_features[feature_category]:
                                        org_features[feature_category].append(key)

                            target_start_id = 0
                            for org_key, info in ordered_origin_keys.items():
                                org_info = info.copy()
                                org_key = org_key.split("*")[0]
                                if org_key not in reverse_convert_features:
                                    reverse_convert_features[org_key] = []
                                if "start" in org_info:
                                    org_info["target_key"] = target_feature
                                    org_info["target_start"] = target_start_id
                                    org_info["target_end"] = target_start_id + org_info["end"] - org_info["start"]
                                    target_start_id = org_info["target_end"]
                                else:
                                    org_info["target_key"] = target_feature
                                reverse_convert_features[org_key].append(org_info)

                        if isinstance(feature_convert_info[target_feature]["origin_keys"], str):
                            to_convert_features[target_feature] = feature_convert_info[target_feature]
                            reverse_convert_features[feature_convert_info[target_feature]["origin_keys"]] = {
                                "target_key": target_feature
                            }
                            origin_key = feature_convert_info[target_feature]["origin_keys"]
                            if origin_key not in org_features[feature_category]:
                                org_features[feature_category].append(origin_key)

        self.action_subtract_state = action_subtract_state
        for feature_category, feature in org_features.items():
            if len(feature) == 0:
                org_features[feature_category] = list(target_features[feature_category])

        self.key_mapping = to_convert_features
        self.key_reverse_mapping = reverse_convert_features

    def convert_features(self, item):
        # Remap a data item from original dataset keys to target feature keys
        # using the forward key_mapping. Supports both simple renaming and
        # slicing + concatenation from multiple origin keys.
        out_item = {}
        for target_key, convert_info in self.key_mapping.items():
            if not self.load_image and target_key in self.images:
                continue
            if isinstance(convert_info["origin_keys"], str) and convert_info["origin_keys"] in item:
                out_item[target_key] = item[convert_info["origin_keys"]]
                continue

            assert isinstance(convert_info["origin_keys"], OrderedDict)
            concat_list = []
            convert_success = True
            for origin_key, origin_info in convert_info["origin_keys"].items():
                origin_key = origin_key.split("*")[0]
                if origin_key not in item:
                    convert_success = False
                    break
                origin_data = item.get(origin_key)[..., origin_info["start"] : origin_info["end"]]
                concat_list.append(origin_data)
            if convert_success:
                out_item[target_key] = torch.cat(concat_list, dim=-1)

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]
        return out_item

    def reverse_features(self, item):
        out_item = {}
        for target_key, convert_info in self.key_reverse_mapping.items():
            if isinstance(convert_info, dict) and convert_info["target_key"] in item:
                out_item[target_key] = item[convert_info["target_key"]]
                continue

            if isinstance(convert_info, list):
                convert_info = sorted(convert_info, key=lambda x: x["end"])
                concat_list = []
                convert_success = True
                for _convert_info in convert_info:
                    if _convert_info["target_key"] not in item:
                        raise ValueError(
                            f"{_convert_info['target_key']} is not contained in robot config as target feature"
                        )

                    concat_list.append(
                        item[_convert_info["target_key"]][
                            ..., _convert_info["target_start"] : _convert_info["target_end"]
                        ]
                    )
                if convert_success:
                    out_item[target_key] = torch.cat(concat_list, dim=-1)

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]
        return out_item

    def reverse_pad_and_concat(self, item):
        # Reverse the pad_and_concat operation: split the concatenated state/action tensors
        # back into per-joint features using the joint mask and normalizer stats for dimensions.
        reverse_item = {}

        joint_mask = item["joint_mask"]
        state = item["state"][joint_mask]
        action = item["actions"][:, joint_mask]

        for k in self.feature_config.joints:
            state_key = f"observation.state.{k}"
            if state_key in self.states:
                joint_dim = self.normalizer.norm_stats[state_key]["mean"].shape[-1]
                reverse_item[state_key] = state[:joint_dim]
                state = state[joint_dim:]

            action_key = f"action.{k}"
            if action_key in self.actions:
                joint_dim = self.normalizer.norm_stats[action_key]["mean"].shape[-1]
                reverse_item[action_key] = action[:, :joint_dim]
                action = action[:, joint_dim:]
        return reverse_item

    def pad_and_concat(self, item):
        # Pad each joint's state and action to its max dimension, then concatenate all joints
        # into unified state/action tensors. Also prepares images and builds a joint_mask
        # indicating which dimensions are real vs padding.
        images = {}
        for image_key in self.feature_config.images:
            if image_key in self.images and image_key in item:
                images[image_key] = (item[image_key] * 255).to(torch.uint8)

        actions, action_joints_pad = [], []
        states, state_joints_pad = [], []
        for k in self.feature_config.joints:
            state_key = f"observation.state.{k}"

            if state_key in self.states:
                pad_len = self.feature_config.joints_max_dim[k] - item[state_key].shape[-1]
                states.append(F.pad(item[state_key], (0, pad_len)))
                state_joints_pad.append(F.pad(torch.ones(item[state_key].shape), (0, pad_len)))
            else:
                states.append(torch.zeros(self.feature_config.joints_max_dim[k]))
                state_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del state_key

            action_key = f"action.{k}"
            if action_key in self.actions:
                pad_len = self.feature_config.joints_max_dim[k] - item[action_key].shape[-1]
                actions.append(F.pad(item[action_key], (0, pad_len)))
                action_joints_pad.append(F.pad(torch.ones(item[action_key].shape[-1]), (0, pad_len)))
            else:
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                action_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del action_key

        batch_dict = {
            "image": images,
            "state": torch.cat(states, dim=-1).to(torch.float32),
            "action": torch.cat(actions, dim=-1).to(torch.float32),
            "action_is_pad": item["action_is_pad"],
            "joint_mask": torch.cat(action_joints_pad, dim=-1).to(dtype=torch.bool),
            "prompt": [item["task"]],
        }

        return batch_dict

    def apply(self, item):
        # Full forward transform pipeline: extract action_is_pad flag, remap features,
        # subtract states from actions if configured, normalize, then pad/concat into
        # model-ready tensors (images, state, actions, language tokens, masks).
        item["action_is_pad"] = item[f"{self.org_features['actions'][0]}_is_pad"]

        item = self.convert_features(item)

        for action_feature in self.actions:
            if self.action_subtract_state[action_feature]:
                state_feature = action_feature.replace("action.", "observation.state.")
                if not (action_feature in item and state_feature in item):
                    raise ValueError(f"{action_feature} or/and {state_feature} are not in the item")

                item[action_feature] -= item[action_feature.replace("action.", "observation.state.")]

        if self.normalizer is not None:
            item = self.normalizer.normalize(item)
        if self.return_item_befor_padding:
            return item

        batch_dict = self.pad_and_concat(item)
        state = prepare_state(batch_dict, self.data_config.max_state_dim)
        actions = prepare_action(batch_dict, self.data_config.max_action_dim)
        joint_mask = prepare_joint_pad(batch_dict, self.data_config.max_action_dim)
        images, img_masks, pil_images = prepare_images(
            self.image_processor,
            batch_dict,
            resize_imgs_with_padding=self.data_config.resize_imgs_with_padding,
            use_depth_align=self.use_depth_align,
            image_keys=self.feature_config.images,
        )

        lang_tokens, lang_masks = prepare_language(self.tokenizer, batch_dict, self.data_config.tokenizer_max_length)
        action_is_pad = batch_dict["action_is_pad"]

        batch_dict = {
            "images": images,
            "img_masks": img_masks,
            "state": state,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "actions": actions,
            "action_is_pad": action_is_pad,
            "joint_mask": joint_mask,
        }
        if self.use_depth_align:
            batch_dict["pil_images"] = pil_images
        return batch_dict

    def unapply(self, item):
        # Inverse of apply(): reverse padding, unnormalize, add back subtracted states,
        # and reverse feature key mapping to recover the original dataset format.
        if not self.return_item_befor_padding:
            item = self.reverse_pad_and_concat(item)

        if self.normalizer is not None:
            item = self.normalizer.unnormalize(item)

        for action_feature in self.actions:
            if self.action_subtract_state[action_feature]:
                item[action_feature] += item[action_feature.replace("action.", "observation.state.")]

        item = self.reverse_features(item)
        return item
