import argparse
import json
import os

import cv2
import decord
import numpy as np
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from utils.curricularface import get_model


def load_image(image):
    img = image.convert("RGB")
    img = transforms.Resize((299, 299))(img)
    img = transforms.ToTensor()(img)
    return img.unsqueeze(0)


def sample_video_frames(video_path, num_frames=32):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if num_frames is None:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        frame = vr[int(idx)].asnumpy()
        frames.append(frame)
    return frames


def get_image_path_from_json(json_data, video_name, input_image_folder):
    video_prefix = video_name.split(".")[0]
    if video_prefix in json_data.keys():  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.
        img_paths = json_data[video_prefix].get("img_paths", [])
        for img_path in img_paths:
            if "/man/" in img_path or "/woman/" in img_path or "/crop_woman/" in img_path or "/crop_man/" in img_path:
                img_path = os.path.join(input_image_folder, img_path)
                return img_path
    return None


def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    return cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)), (
        left,
        top,
    )


def get_face_keypoints(face_model, image_bgr):
    face_info = face_model.get(image_bgr)
    if len(face_info) > 0:
        return sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[-1]
    return None


def batch_cosine_similarity(embedding_image, embedding_frames, device="cuda"):
    embedding_image = torch.tensor(embedding_image).to(device)
    embedding_frames = torch.tensor(embedding_frames).to(device)
    return torch.nn.functional.cosine_similarity(embedding_image, embedding_frames, dim=-1).cpu().numpy()


@torch.no_grad()
def inference(face_model, img, device):
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    embedding = face_model(img).detach().cpu().numpy()[0]
    return embedding / np.linalg.norm(embedding)


def process_image(face_model, image_path):
    if isinstance(image_path, str):
        np_faceid_image = np.array(Image.open(image_path).convert("RGB"))
    elif isinstance(image_path, np.ndarray):
        np_faceid_image = image_path
    else:
        raise TypeError("image_path should be a string or PIL.Image.Image object")

    image_bgr = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)

    face_info = get_face_keypoints(face_model, image_bgr)
    if face_info is None:
        padded_image, sub_coord = pad_np_bgr_image(image_bgr)
        face_info = get_face_keypoints(face_model, padded_image)
        if face_info is None:
            print("Warning: No face detected in the image. Continuing processing...")
            return None, None
        face_kps = face_info["kps"]
        face_kps -= np.array(sub_coord)
    else:
        face_kps = face_info["kps"]
    arcface_embedding = face_info["embedding"]

    norm_face = face_align.norm_crop(image_bgr, landmark=face_kps, image_size=224)
    align_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB)

    return align_face, arcface_embedding


def process_video(
    video_path,
    face_arc_model,
    face_cur_model,
    arcface_image_embedding,
    cur_image_embedding,
    device,
    num_frames=32,
):
    video_frames = sample_video_frames(video_path, num_frames=num_frames)
    cur_scores = []
    arc_scores = []
    for frame_rgb in video_frames:
        align_face_frame, arcface_frame_embedding = process_image(face_arc_model, frame_rgb)
        if align_face_frame is None:
            continue
        cur_embedding_frame = inference(face_cur_model, align_face_frame, device)
        cur_score = max(
            0.0,
            batch_cosine_similarity(cur_image_embedding, cur_embedding_frame, device=device).item(),
        )
        arc_score = max(
            0.0,
            batch_cosine_similarity(arcface_image_embedding, arcface_frame_embedding, device=device).item(),
        )
        if cur_score != 0.0:
            cur_scores.append(cur_score)
        if arc_scores != 0.0:
            arc_scores.append(arc_score)
    avg_cur_score = np.mean(cur_scores) if cur_scores else 0.0
    avg_arc_score = np.mean(arc_scores) if arc_scores else 0.0
    return avg_cur_score, avg_arc_score


def process_video_files(
    input_video_folder,
    input_json_file,
    face_arc_model,
    face_cur_model,
    input_image_folder,
    device="cuda",
    num_frames=32,
):
    with open(input_json_file) as f:
        json_data = json.load(f)
    video_files = [f for f in os.listdir(input_video_folder) if f.endswith(".mp4")]

    results = {}
    for video_file in tqdm(video_files):
        print(f"Processing video: {video_file}")
        file_basename = os.path.basename(video_file)
        if not any(keyword in file_basename for keyword in ["singlehuman", "singleface", "faceobj", "humanobj"]):
            continue

        # For image
        image_path = get_image_path_from_json(json_data, video_file, input_image_folder)
        if image_path is None:
            print(f"Error: No valid image path found for {video_file}")
            continue
        align_face_image, arcface_image_embedding = process_image(face_arc_model, image_path)
        if align_face_image is None:
            print(f"Error processing image at {image_path}")
            continue
        cur_image_embedding = inference(face_cur_model, align_face_image, device)

        # For video
        cur_score, arc_score = process_video(
            video_path=os.path.join(input_video_folder, video_file),
            face_arc_model=face_arc_model,
            face_cur_model=face_cur_model,
            arcface_image_embedding=arcface_image_embedding,
            cur_image_embedding=cur_image_embedding,
            device=device,
            num_frames=num_frames,
        )

        video_prefix = video_file.split(".")[0]
        results[video_prefix] = {"cur_score": cur_score, "arc_score": arc_score}
        print(f"cur score: {cur_score}")
        print(f"arc score: {arc_score}")

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video_folder",
        type=str,
        default="demo_result/model_name_input_video",
    )
    parser.add_argument(
        "--input_image_folder",
        type=str,
        default="BestWishYsh/OpenS2V-Eval",
    )
    parser.add_argument(
        "--input_json_file",
        type=str,
        default="demo_result/input_json/Human-Domain_Eval.json",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        default="demo_result/model_name_output_json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="BestWishYsh/OpenS2V-Weight",
    )
    parser.add_argument("--num_frames", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    input_video_folder = args.input_video_folder
    input_image_folder = args.input_image_folder
    input_json_file = args.input_json_file
    output_json_folder = args.output_json_folder
    model_path = args.model_path
    num_frames = args.num_frames

    output_json_file = os.path.join(output_json_folder, "facesim.json")
    os.makedirs(output_json_folder, exist_ok=True)
    if os.path.exists(output_json_file):
        print("continue")
        return

    face_arc_path = os.path.join(model_path, "face_extractor")
    face_cur_path = os.path.join(model_path, "glint360k_curricular_face_r101_backbone.bin")

    face_arc_model = FaceAnalysis(root=face_arc_path, providers=["CUDAExecutionProvider"])
    face_arc_model.prepare(ctx_id=0, det_size=(320, 320))

    face_cur_model = get_model("IR_101")([112, 112])
    face_cur_model.load_state_dict(torch.load(face_cur_path, map_location="cpu"))
    face_cur_model = face_cur_model.to(device)
    face_cur_model.eval()

    results = process_video_files(
        input_video_folder,
        input_json_file,
        face_arc_model,
        face_cur_model,
        input_image_folder,
        device=device,
        num_frames=num_frames,
    )

    with open(output_json_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"All results have been saved to {output_json_file}")


if __name__ == "__main__":
    main()
