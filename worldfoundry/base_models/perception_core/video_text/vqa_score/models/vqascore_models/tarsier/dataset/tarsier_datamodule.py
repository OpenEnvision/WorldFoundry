"""Datamodule for Llava Pretraining and Finetuning"""

import json
import traceback

import torch
from PIL import Image
from transformers import DataCollatorForSeq2Seq

from ..tools.rw_utils import NumpyArrayEncoder
from .custom_data_parsers.standard_vision_parser import VisionParser
from .tarsier_processor import TarsierProcessor
from .utils import DictToObject


class TarsierDataProcessor:
    def __init__(
        self,
        processor: TarsierProcessor,
        n_frames: int | list,
        max_n_frames=256,
        max_pixels=1280 * 720 // 2,
        min_pixels=0,
        max_seq_len=None,
        is_training=True,  # Controls frame sampling and omits the response during evaluation.
        print_data_error=True,
        do_image_padding=False,
        do_image_crop=False,
        do_image_resize=True,
        video_sampling_strategy={},  # noqa: B006 - Keeps the upstream default object and signature.
        prompt="",
        train_task="sft",
        **kwargs,
    ):
        self.kwargs = kwargs

        self.processor = processor
        self.pad_collator = DataCollatorForSeq2Seq(processor.tokenizer, padding="longest")

        self.processor.max_seq_len = self.tokenizer.model_max_length if max_seq_len is None else max_seq_len

        self.n_frames = n_frames
        self.max_n_frames = max_n_frames
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        self.is_training = is_training
        self.print_data_error = print_data_error
        self.do_image_padding = do_image_padding
        self.do_image_crop = do_image_crop
        self.do_image_resize = do_image_resize
        self.video_sampling_strategy = video_sampling_strategy
        self.prompt = prompt
        self.train_task = train_task

        self.vision_parser = VisionParser(
            n_frames=self.n_frames,
            max_n_frames=self.max_n_frames,
            is_training=self.is_training,
            video_sampling_strategy=self.video_sampling_strategy,
        )

    def select_parser(self, data_dict):
        return self.vision_parser

    def parse_image_processing_config(self, data_dict):
        image_processing_config = data_dict.get("image_processing_config", {})

        do_padding = image_processing_config.get("do_padding", self.do_image_padding)
        do_crop = image_processing_config.get("do_crop", self.do_image_crop)
        do_resize = image_processing_config.get("do_resize", self.do_image_resize)
        max_pixels = image_processing_config.get("max_pixels", self.max_pixels)
        min_pixels = image_processing_config.get("min_pixels", self.min_pixels)

        assert min_pixels <= max_pixels

        image_processing_config["do_padding"] = do_padding
        image_processing_config["do_crop"] = do_crop
        image_processing_config["do_resize"] = do_resize
        image_processing_config["max_pixels"] = max_pixels
        image_processing_config["min_pixels"] = min_pixels

        return image_processing_config

    def _transform(self, raw_data_dict: dict) -> dict:
        data_dict = json.loads(json.dumps(raw_data_dict, cls=NumpyArrayEncoder))
        del raw_data_dict

        if self.prompt:
            for msg in data_dict["messages"]:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "text":
                            content["text"] = self.prompt

        data_dict_copy = json.loads(json.dumps(data_dict, cls=NumpyArrayEncoder))
        data_dict.setdefault("extra_info", {})

        image_processing_config = self.parse_image_processing_config(data_dict)
        parser = self.select_parser(data_dict)
        messages = parser.transform(data_dict, image_processing_config)
        data_dict_copy["extra_info"] = data_dict.pop("extra_info", {})

        outputs = self.processor(messages, image_processing_config, is_training=self.is_training)

        # if not self.is_training:
        outputs["raw_data_dict"] = data_dict_copy

        return [outputs]

    def _split_chosen_rejected(self, data_dict: dict):
        chosen_data_dict = data_dict
        rejected_data_dict = json.loads(json.dumps(data_dict, cls=NumpyArrayEncoder))
        for msg in chosen_data_dict["messages"]:
            if msg["role"] == "assistant":
                for content in msg["content"]:
                    if content["type"] == "text":
                        content["text"] = content["chosen"]

        for msg in rejected_data_dict["messages"]:
            if msg["role"] == "assistant":
                for content in msg["content"]:
                    if content["type"] == "text":
                        content["text"] = content["rejected"]

        return chosen_data_dict, rejected_data_dict

    def transform(self, data_dict: dict) -> dict:
        try:
            if self.train_task == "dpo":
                chosen_data_dict, rejected_data_dict = self._split_chosen_rejected(data_dict)
                return self._transform(chosen_data_dict) + self._transform(rejected_data_dict)
            return self._transform(data_dict)
        except Exception:
            if self.print_data_error:
                print(traceback.format_exc())
                print(f"Error occurs when processing: \n{data_dict}")
            return []

    def batch_transform(self, batch_data: list[dict]) -> dict:
        model_inputs = {}
        # if not self.is_training:
        raw_data_dict = [d.pop("raw_data_dict") for d in batch_data]
        model_inputs["raw_data_dict"] = raw_data_dict

        batch_pixel_values = [d.pop("pixel_values") for d in batch_data if "pixel_values" in d]
        batch_image_grid_thw = [d.pop("image_grid_thw") for d in batch_data if "image_grid_thw" in d]
        if len(batch_pixel_values) == 0:
            vision_placeholder = self.get_vision_placeholder()
            batch_pixel_values = [vision_placeholder.get("pixel_values")]
            batch_image_grid_thw = (
                [vision_placeholder.get("image_grid_thw")] if "image_grid_thw" in vision_placeholder else []
            )

        model_inputs["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
        if len(batch_image_grid_thw) > 0:
            model_inputs["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)

        batch_num_images = [d.pop("num_images") for d in batch_data]
        model_inputs["num_images"] = torch.tensor(batch_num_images)
        model_inputs.update(self.pad_collator(batch_data))
        return model_inputs

    def __call__(self, batch_data: dict | list[dict]) -> dict:
        if isinstance(batch_data, dict):
            batch_data = [batch_data]
        batch = [self.transform(d)[0] for d in batch_data]
        return self.batch_transform(batch)

    def get_vision_placeholder(self):
        messages = [{"role": "user", "content": [{"type": "image", "image": Image.new(mode="RGB", size=(336, 336))}]}]
        image_processing_config = self.parse_image_processing_config({})
        return self.processor(messages, image_processing_config)

    def get_text_placeholder(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Thank you very much"}]},
        ]
        image_processing_config = self.parse_image_processing_config({})
        return self.processor(messages, image_processing_config)


def init_processor(processor: TarsierProcessor | str = None, config: dict | None = None):
    config = DictToObject(config) if isinstance(config, dict) else config
    if isinstance(processor, str):
        sub_processor = TarsierProcessor.from_pretrained(processor, padding_side="left", trust_remote_code=True)
    else:
        sub_processor = processor
    processor = TarsierDataProcessor(
        processor=sub_processor,
        n_frames=config.n_frames,
        max_n_frames=config.max_n_frames,
        max_pixels=config.max_pixels,
        min_pixels=config.min_pixels,
        max_seq_len=config.max_seq_len,
        is_training=config.is_training,
        print_data_error=config.print_data_error,
        do_image_padding=config.do_image_padding,
        do_image_crop=config.do_image_crop,
        do_image_resize=config.do_image_resize,
        video_sampling_strategy=config.video_sampling_strategy,
        prompt=config.prompt,
        train_task=config.train_task,
    )
    return processor
