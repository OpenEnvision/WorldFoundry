# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Local HunyuanVideo 1.5 text, glyph and vision resource loading."""

import json
import os

import loguru
from loguru import logger

from .h15_text import PROMPT_TEMPLATE, TextEncoder
from .h15_text.byT5 import load_glyph_byT5_v2
from .h15_text.byT5.format_prompt import MultilingualPromptFormat
from .h15_vision import VisionEncoder


class HunyuanVideo15Resources:
    @classmethod
    def _empty_byt5_kwargs(cls, byt5_max_length):
        return {"byt5_model": None, "byt5_tokenizer": None, "byt5_max_length": byt5_max_length}

    @classmethod
    def _local_byt5_candidates(cls, load_from):
        ckpt_root = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
        candidates = [os.path.join(load_from, "byt5-small")]
        if ckpt_root:
            candidates.extend(
                [os.path.join(ckpt_root, "byt5-small"), os.path.join(ckpt_root, "hfd", "google--byt5-small")]
            )
        return candidates

    @classmethod
    def _load_byt5(cls, cached_folder, glyph_byT5_v2, byt5_max_length, device):
        if not glyph_byT5_v2:
            byt5_kwargs = None
            prompt_format = None
            return (byt5_kwargs, prompt_format)
        try:
            load_from = os.path.join(cached_folder, "text_encoder")
            glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
            required_glyph_files = [
                os.path.join(glyph_root, "assets/color_idx.json"),
                os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json"),
                os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
            ]
            missing_glyph_files = [path for path in required_glyph_files if not os.path.exists(path)]
            if missing_glyph_files:
                loguru.logger.warning(
                    "Glyph-SDXL-v2 checkpoint is incomplete under {}. Plain prompts will use zero ByT5 glyph embeddings; prompts with quoted glyph text still require Glyph-SDXL-v2. Missing: {}",
                    glyph_root,
                    missing_glyph_files,
                )
                return (cls._empty_byt5_kwargs(byt5_max_length), None)
            byT5_google_path = next(
                (candidate for candidate in cls._local_byt5_candidates(load_from) if os.path.exists(candidate)), None
            )
            if byT5_google_path is None:
                loguru.logger.warning(
                    f"ByT5 google path not found from: {load_from}. Try downloading from https://huggingface.co/google/byt5-small."
                )
                byT5_google_path = "google/byt5-small"
            multilingual_prompt_format_color_path = os.path.join(glyph_root, "assets/color_idx.json")
            multilingual_prompt_format_font_path = os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json")
            byt5_args = dict(
                byT5_google_path=byT5_google_path,
                byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
                multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
                multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
                byt5_max_length=byt5_max_length,
            )
            byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
            prompt_format = MultilingualPromptFormat(
                font_path=multilingual_prompt_format_font_path, color_path=multilingual_prompt_format_color_path
            )
            return (byt5_kwargs, prompt_format)
        except Exception as e:
            raise RuntimeError("Error loading byT5 glyph processor") from e

    @classmethod
    def _load_text_encoders(cls, pretrained_model_path, device):
        text_encoder_path = None
        text_encoder_override = os.environ.get("HUNYUANWORLDPLAY_TEXT_ENCODER_PATH")
        ckpt_root = os.environ.get("WORLDFOUNDRY_CKPT_DIR") or os.path.dirname(pretrained_model_path)
        expected_hidden_size = 3584
        candidates = [
            text_encoder_override,
            f"{pretrained_model_path}/text_encoder/llm",
            f"{pretrained_model_path}/text_encoder",
            os.path.join(ckpt_root, "Qwen2.5-VL-7B-Instruct"),
            os.path.join(ckpt_root, "Qwen", "Qwen2.5-VL-7B-Instruct"),
            os.path.join(ckpt_root, "hfd", "Qwen--Qwen2.5-VL-7B-Instruct"),
            os.path.join(ckpt_root, "HunyuanVideo", "text_encoder", "llm"),
            os.path.join(ckpt_root, "HunyuanVideo", "text_encoder"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            config_path = os.path.join(candidate, "config.json")
            if not os.path.exists(config_path):
                continue
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as exc:
                loguru.logger.warning("Skipping unreadable HY-WorldPlay text encoder config {}: {}", config_path, exc)
                continue
            hidden_size = config.get("hidden_size")
            if hidden_size is None and isinstance(config.get("text_config"), dict):
                hidden_size = config["text_config"].get("hidden_size")
            if hidden_size != expected_hidden_size:
                loguru.logger.warning(
                    "Skipping HY-WorldPlay text encoder candidate {}: hidden_size={} but expected {}.",
                    candidate,
                    hidden_size,
                    expected_hidden_size,
                )
                continue
            text_encoder_path = candidate
            break
        if not text_encoder_path or not os.path.exists(text_encoder_path):
            msg = f"HY-WorldPlay text encoder not found. Checked: {[candidate for candidate in candidates if candidate]}. Expected a Qwen2.5-VL-compatible encoder with hidden_size={expected_hidden_size}. Please refer to checkpoints-download.md."
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)
        text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=text_encoder_path,
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=loguru.logger,
            device=device,
        )
        text_encoder_2 = None
        return (text_encoder, text_encoder_2)

    @classmethod
    def _load_vision_encoder(cls, pretrained_model_name_or_path, device):
        vision_encoder_path = None
        vision_encoder_override = os.environ.get("HUNYUANWORLDPLAY_VISION_ENCODER_PATH")
        ckpt_root = os.environ.get("WORLDFOUNDRY_CKPT_DIR") or os.path.dirname(pretrained_model_name_or_path)
        expected_hidden_size = 1152
        candidates = [
            vision_encoder_override,
            f"{pretrained_model_name_or_path}/vision_encoder/siglip",
            os.path.join(ckpt_root, "FLUX.1-Redux-dev"),
            os.path.join(ckpt_root, "black-forest-labs--FLUX.1-Redux-dev"),
            os.path.join(ckpt_root, "hfd", "black-forest-labs--FLUX.1-Redux-dev"),
            os.path.join(ckpt_root, "siglip-base-patch16-224"),
            os.path.join(ckpt_root, "google--siglip-base-patch16-224"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            config_path = os.path.join(candidate, "image_encoder", "config.json")
            if not os.path.exists(config_path):
                config_path = os.path.join(candidate, "config.json")
            if not os.path.exists(config_path):
                continue
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as exc:
                loguru.logger.warning("Skipping unreadable HY-WorldPlay vision encoder config {}: {}", config_path, exc)
                continue
            hidden_size = config.get("hidden_size")
            if hidden_size is None and isinstance(config.get("vision_config"), dict):
                hidden_size = config["vision_config"].get("hidden_size")
            if hidden_size != expected_hidden_size:
                loguru.logger.warning(
                    "Skipping HY-WorldPlay vision encoder candidate {}: hidden_size={} but expected {}.",
                    candidate,
                    hidden_size,
                    expected_hidden_size,
                )
                continue
            vision_encoder_path = candidate
            break
        if not vision_encoder_path or not os.path.exists(vision_encoder_path):
            msg = f"HY-WorldPlay vision encoder not found. Checked: {[candidate for candidate in candidates if candidate]}. Expected a FLUX.1-Redux/SigLIP-compatible encoder with hidden_size={expected_hidden_size}. Please refer to checkpoints-download.md."
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)
        vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=vision_encoder_path,
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=logger,
            device=device,
        )
        return vision_encoder
