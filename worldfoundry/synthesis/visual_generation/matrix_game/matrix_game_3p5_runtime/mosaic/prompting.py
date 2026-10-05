"""Prompt validation for Matrix-Game inference."""


def _prompt_context_from_data(data):
    if not isinstance(data, dict):
        return ""
    clean_name = (data.get("info") or {}).get("clean_name")
    return f" clean_name={clean_name}" if clean_name else ""


def _require_nonempty_prompt(
    prompt,
    *,
    phase,
    data=None,
    clean_name=None,
    allow_empty=False,
):
    text = str(prompt or "").strip()
    if text:
        return text
    context = f" clean_name={clean_name}" if clean_name is not None else _prompt_context_from_data(data)
    if not allow_empty:
        raise ValueError(f"{phase} received empty prompt{context}.")
    warned = getattr(_require_nonempty_prompt, "_warned_empty", set())
    key = (str(phase), context)
    if key not in warned and len(warned) < 32:
        warned.add(key)
        _require_nonempty_prompt._warned_empty = warned
        print(
            f"[prompt] WARNING: {phase} received empty prompt{context}; continuing because empty prompts are enabled.",
            flush=True,
        )
    return ""


__all__ = ["_require_nonempty_prompt"]
