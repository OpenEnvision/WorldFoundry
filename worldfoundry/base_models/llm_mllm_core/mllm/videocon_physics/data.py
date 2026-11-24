from __future__ import annotations

import re

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


class MultiModalDataset(Dataset):
    def __init__(
        self,
        input_file: str,
        tokenizer,
        processor,
        *,
        max_length: int = 256,
        num_frames: int = 32,
        media_tokens: tuple[str, str] = ("<image>", "<|video|>"),
    ) -> None:
        self.dataset = pd.read_csv(input_file).dropna()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_frames = num_frames
        self.processor = processor
        self.media_tokens = {token: -int(index + 1) for index, token in enumerate(media_tokens)}
        self.media_lengths = {"<image>": 65, "<|video|>": 65}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        data = self.dataset.iloc[index]
        videopath = str(data["videopath"])
        caption = str(data["caption"])
        video_input = self.processor(videos=[videopath], num_frames=self.num_frames, return_tensors="pt")
        text_input = self._extract_text_token_from_conversation(caption, self.max_length, index)
        return {"video": video_input, "text": text_input, "videopath": videopath, "caption": caption}

    def _extract_text_token_from_conversation(
        self,
        conversation: str,
        max_length: int,
        index: int,
    ) -> dict[str, torch.Tensor]:
        enc_chunk: list[int] = []
        label_chunk: list[int] = []
        prompt_chunk = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id > 0 else []

        if all(media_token not in conversation for media_token in self.media_tokens):
            pattern = "|".join(map(re.escape, ["AI: ", "\nHuman: "]))
            chunk_strs = re.split(f"({pattern})", conversation)
            enc_length = 0
            stop_flag = False
            for chunk_index, chunk_str in enumerate(chunk_strs):
                if chunk_index == 0:
                    enc_chunk = prompt_chunk + self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                    continue
                curr_chunk = self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                if chunk_strs[chunk_index - 1] == "AI: ":
                    if enc_length + len(curr_chunk) >= max_length:
                        curr_chunk = curr_chunk[: max_length - enc_length]
                        stop_flag = True
                    curr_chunk += [self.tokenizer.eos_token_id]
                    label_value = 1
                else:
                    if enc_length + len(curr_chunk) >= max_length + 1:
                        curr_chunk = curr_chunk[: max_length + 1 - enc_length]
                        stop_flag = True
                    label_value = 0
                enc_length += len(curr_chunk)
                enc_chunk += curr_chunk
                label_chunk += [label_value] * len(curr_chunk)
                if stop_flag:
                    break
        else:
            enc_length = 0
            pattern = "|".join(map(re.escape, [*self.media_tokens.keys(), "AI: ", "\nHuman: "]))
            chunk_strs = [chunk for chunk in re.split(f"({pattern})", conversation) if chunk]
            for chunk_index, chunk_str in enumerate(chunk_strs):
                if enc_length >= max_length + 1:
                    break
                if chunk_index == 0:
                    enc_chunk = prompt_chunk + self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                    continue
                if chunk_str in self.media_tokens:
                    media_length = self.media_lengths[chunk_str]
                    if enc_length + media_length > max_length + 1:
                        break
                    enc_chunk += [self.media_tokens[chunk_str]] * media_length
                    enc_length += media_length
                    label_chunk += [0] * media_length
                    continue
                curr_chunk = self.tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                if chunk_strs[chunk_index - 1] == "AI: ":
                    if enc_length + len(curr_chunk) >= max_length:
                        curr_chunk = curr_chunk[: max_length - enc_length]
                    curr_chunk += [self.tokenizer.eos_token_id]
                    label_value = 1
                else:
                    if enc_length + len(curr_chunk) >= max_length + 1:
                        curr_chunk = curr_chunk[: max_length + 1 - enc_length]
                    label_value = 0
                enc_length += len(curr_chunk)
                enc_chunk += curr_chunk
                label_chunk += [label_value] * len(curr_chunk)

        if enc_length < max_length + 1:
            padding_length = max_length + 1 - enc_length
            enc_chunk += [self.tokenizer.pad_token_id] * padding_length
            label_chunk += [0] * padding_length
        else:
            padding_length = 0

        assert enc_length + padding_length == max_length + 1, (
            index,
            enc_length,
            padding_length,
            max_length + 1,
        )
        assert len(label_chunk) == max_length + 1, (len(label_chunk), max_length + 1)

        input_ids = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor([1 if i < enc_length - 1 else 0 for i in range(max_length)]).long()
        prompt_mask = torch.tensor(label_chunk)[1:].long()
        if all(media_token not in conversation for media_token in self.media_tokens):
            non_media_mask = torch.ones_like(non_padding_mask).long()
        else:
            tmp_enc_chunk = input_ids.clone()
            tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
            tmp_enc_chunk[tmp_enc_chunk < 0] = 0
            non_media_mask = tmp_enc_chunk[1:].long()
        return {
            "input_ids": input_ids,
            "non_padding_mask": non_padding_mask,
            "non_media_mask": non_media_mask,
            "prompt_mask": prompt_mask,
        }


def batchify(batch: list[dict[str, object]]) -> dict[str, object]:
    videos = [data["video"] for data in batch if data["video"] is not None]
    video = torch.cat(videos, dim=0) if videos else None
    num_videos_per_sample = torch.LongTensor(
        [data["video"].size(0) if data["video"] is not None else 0 for data in batch]
    )
    text = torch.stack([torch.LongTensor(data["text"]["input_ids"]) for data in batch], dim=0)
    non_padding_mask = torch.stack([torch.LongTensor(data["text"]["non_padding_mask"]) for data in batch], dim=0)
    non_media_mask = torch.stack([torch.LongTensor(data["text"]["non_media_mask"]) for data in batch], dim=0)
    prompt_mask = torch.stack([torch.LongTensor(data["text"]["prompt_mask"]) for data in batch], dim=0)
    return {
        "pixel_values": None,
        "video_pixel_values": video,
        "input_ids": text.long(),
        "labels": text.long().clone(),
        "num_images": torch.LongTensor([0 for _ in batch]).long(),
        "num_videos": num_videos_per_sample.long(),
        "non_padding_mask": non_padding_mask.long(),
        "non_media_mask": non_media_mask.long(),
        "prompt_mask": prompt_mask.long(),
        "videopaths": [str(data["videopath"]) for data in batch],
        "captions": [str(data["caption"]) for data in batch],
    }
