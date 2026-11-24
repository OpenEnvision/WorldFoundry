PROMPT_SA = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    'Human: Does this video match the description: "{caption}"? '
    "Please rate the video on a scale from 1 to 5, where 5 indicates a perfect match and 1 indicates no relevance.\n"
    "AI: "
)

PROMPT_PHYSICS = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does this video adhere to the physical laws? "
    "Rate the video on a scale from 1 to 5, where 5 means full compliance and 1 means significant violations.\n"
    "AI: "
)

PROMPT_RULE = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    'Human: Does the video follow the physical rule: "{rule}"?\n'
    "Choose 0 if not, 1 if valid, or 2 if indeterminate. \n"
    "AI: "
)
