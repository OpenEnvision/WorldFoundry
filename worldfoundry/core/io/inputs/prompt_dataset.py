"""Small line-oriented prompt datasets shared by inference runtimes.

Eval and batch-generate jobs often store one prompt per line, with an
optional second file of "extended" prompts (rewrites, negatives) that
must stay aligned. :class:`TextPromptDataset` is a torch
``Dataset`` over those files so a DataLoader can feed official
benchmarks without a custom collate.

The two files must have the same line count when the extended path is
set. Encoding is UTF-8. This is not a webdataset and not a JSONL
conversation loader.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────
# Line-aligned prompt pairs — UTF-8; not webdataset / JSONL conversations
# ──────────────────────────────────────────────────────────────────────────


class TextPromptDataset(Dataset):
    """Read one prompt per line with an optional aligned extended-prompt file."""

    def __init__(self, prompt_path: str | Path, extended_prompt_path: str | Path | None = None) -> None:
        """Load UTF-8 lines; raise if the optional extended file has a different length."""

        self.prompt_list = Path(prompt_path).read_text(encoding="utf-8").splitlines()
        if extended_prompt_path is None:
            self.extended_prompt_list = None
        else:
            self.extended_prompt_list = Path(extended_prompt_path).read_text(encoding="utf-8").splitlines()
            if len(self.extended_prompt_list) != len(self.prompt_list):
                raise ValueError(
                    "Prompt and extended-prompt files must contain the same number of lines: "
                    f"{len(self.prompt_list)} != {len(self.extended_prompt_list)}"
                )

    def __len__(self) -> int:
        """Number of prompt lines (and extended lines, when present)."""

        return len(self.prompt_list)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return ``prompts`` / ``idx`` and optional ``extended_prompts`` for one row."""

        sample: dict[str, object] = {"prompts": self.prompt_list[index], "idx": index}
        if self.extended_prompt_list is not None:
            sample["extended_prompts"] = self.extended_prompt_list[index]
        return sample


__all__ = ["TextPromptDataset"]
