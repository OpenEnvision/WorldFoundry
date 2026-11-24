import os

VALID_DATA_FORMAT_STRING = "Input data must be {'.jpg', '.jpeg', '.png', '.tif'} for image; or {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv', '.gif'}  for videos!"


def get_visual_type(input_file):
    ext = os.path.splitext(input_file)[-1]
    if ext in {".gif"}:
        return "gif"
    elif ext in {".mp4", ".avi", ".webm", ".mov", ".mkv", ".wmv"}:
        return "video"
    elif ext in {".jpg", ".jpeg", ".png", ".tif"}:
        return "image"
    else:
        print(f"{VALID_DATA_FORMAT_STRING} But found {ext}!")
        return "unk"


def format_one_sample(media_file=None, prompt="Describe the video in detail."):
    sample = {"messages": []}
    user_content = {"role": "user", "content": []}
    if media_file is not None:
        media_type = get_visual_type(media_file)
        if media_type in ("video", "gif"):
            media_type = "video"
        media_path_key = f"{media_type}_file"
        user_content["content"].append(
            {
                "type": media_type,
                media_type: {
                    media_path_key: media_file,
                },
            }
        )
    user_content["content"].append({"type": "text", "text": prompt})

    assistant_content = {"role": "assistant", "content": []}

    sample["messages"].append(user_content)
    sample["messages"].append(assistant_content)
    if media_file is not None:
        sample["task"] = f"{media_type}/QA"
    else:
        sample["task"] = "text-only"
    return sample


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
