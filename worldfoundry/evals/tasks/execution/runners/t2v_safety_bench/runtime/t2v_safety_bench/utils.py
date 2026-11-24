import base64
import io

import cv2
from PIL import Image


def ext_frame(video_path, num_frames=5, scale_percent=20):
    video = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        frame_number = int((i / num_frames) * total_frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if ret:
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            new_size = (width, height)
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = base64.b64encode(buffer).decode("utf-8")
            frames.append(frame_bytes)

    video.release()
    return frames


def ext_img(file_path):
    frames = []
    image = cv2.imread(file_path)
    _, buffer = cv2.imencode(".jpg", image)
    frame_bytes = base64.b64encode(buffer).decode("utf-8")
    frames.append(frame_bytes)
    return frames


def load_prompts(prompt_path):
    with open(prompt_path, encoding="utf-8", errors="replace") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def merge_args(cfg, args, training=False):
    if args.ckpt_path is not None:
        cfg.model["from_pretrained"] = args.ckpt_path
        args.ckpt_path = None
    if training and args.data_path is not None:
        cfg.dataset["data_path"] = args.data_path
        args.data_path = None
    if not training and args.cfg_scale is not None:
        cfg.scheduler["cfg_scale"] = args.cfg_scale
        args.cfg_scale = None
    if not training and args.num_sampling_steps is not None:
        cfg.scheduler["num_sampling_steps"] = args.num_sampling_steps
        args.num_sampling_steps = None

    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    if not training:
        # Inference only
        # - Allow not set
        if "reference_path" not in cfg:
            cfg["reference_path"] = None
        if "loop" not in cfg:
            cfg["loop"] = 1
        if "frame_interval" not in cfg:
            cfg["frame_interval"] = 1
        if "sample_name" not in cfg:
            cfg["sample_name"] = None
        if "num_sample" not in cfg:
            cfg["num_sample"] = 1
        if "prompt_as_path" not in cfg:
            cfg["prompt_as_path"] = False
        # - Prompt handling
        if "prompt" not in cfg or cfg["prompt"] is None:
            assert cfg["prompt_path"] is not None, "prompt or prompt_path must be provided"
            cfg["prompt"] = load_prompts(cfg["prompt_path"])
        if args.start_index is not None and args.end_index is not None:
            cfg["prompt"] = cfg["prompt"][args.start_index : args.end_index]
        elif args.start_index is not None:
            cfg["prompt"] = cfg["prompt"][args.start_index :]
        elif args.end_index is not None:
            cfg["prompt"] = cfg["prompt"][: args.end_index]
    else:
        # Training only
        # - Allow not set
        if "mask_ratios" not in cfg:
            cfg["mask_ratios"] = None
        if "start_from_scratch" not in cfg:
            cfg["start_from_scratch"] = False
        if "bucket_config" not in cfg:
            cfg["bucket_config"] = None
        if "transform_name" not in cfg.dataset:
            cfg.dataset["transform_name"] = "center"
        if "num_bucket_build_workers" not in cfg:
            cfg["num_bucket_build_workers"] = 1

    # Both training and inference
    if "multi_resolution" not in cfg:
        cfg["multi_resolution"] = False

    return cfg


class ImageProcessor:
    @staticmethod
    def encode_image_from_pil(img):
        buffer = io.BytesIO()
        if img.width > 400 or img.height > 400:
            if img.width > img.height:
                new_width = 400
                concat = float(new_width / float(img.width))
                size = int(float(img.height) * float(concat))
                img = img.resize((new_width, size), Image.LANCZOS)
            else:
                new_height = 400
                concat = float(new_height / float(img.height))
                size = int(float(img.width) * float(concat))
                img = img.resize((size, new_height), Image.LANCZOS)
        img.save(buffer, format="JPEG")
        img_data = base64.b64encode(buffer.getvalue())
        return img_data.decode("utf-8")


def ext_frame_with_encoding(video_path, num_frames=10, scale_percent=50):
    video = cv2.VideoCapture(video_path)
    frames_base64 = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        frame_number = int((i / num_frames) * total_frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if ret:
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            new_size = (width, height)
            frame_resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

            img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            encoded_frame = ImageProcessor.encode_image_from_pil(img_pil)
            frames_base64.append(encoded_frame)

    video.release()
    return frames_base64
