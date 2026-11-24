"""Event prompts reach the released model through its main caption."""

from __future__ import annotations

from collections.abc import Sequence


def merge_event_into_captions(
    text_prompts: Sequence[str],
    event_prompts: str | Sequence[str] | None,
) -> list[str]:
    """Append each event to its caption so open-domain events use Wan's caption prior."""
    captions = list(text_prompts)
    if not event_prompts:
        return captions
    events = [event_prompts] if isinstance(event_prompts, str) else list(event_prompts)
    if len(events) == 1 and len(captions) > 1:
        events *= len(captions)
    if len(events) != len(captions):
        raise ValueError(
            f"{len(events)} event prompts for {len(captions)} captions; "
            "pass one per row, or a single string to apply to all"
        )
    return [
        f"{caption.rstrip()} {event.strip()}" if event and event.strip() else caption
        for caption, event in zip(captions, events, strict=False)
    ]
