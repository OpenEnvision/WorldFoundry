from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from PIL import Image
import numpy as np
from einops import rearrange
from metrics.clip_score import calculate_clip_score
from utils import load_video, load_image
from skimage.metrics import mean_squared_error, structural_similarity as ssim

model, preprocess = clip.load("path_to_dir/ViT-L-14.pt", device="cuda")


# GenVideo-RefVideo SSIM
def compute_video_video_similarity_ssim(reference_video_path,
                                        video_path,
                                        keyframes=[0, 4, 8, 12, 15]):
    global model, preprocess

    reference_video = load_video(reference_video_path)
    try:
        video_frames = load_video(video_path,
                                  size=(reference_video[0].shape[1],
                                        reference_video[0].shape[0]))[:16]
    except:
        print(video_path, ' Error !!')
        return {'ssim': 0, 'state': 0}

    all_frame_scores = 0.
    for idx, frames in enumerate(keyframes):
        im1 = np.array(
            reference_video[10 + frames * 3]
        )  # In webvid, we start with the tenth frame of the video for the motion. Different fps * 3
        im2 = np.array(video_frames[frames])

        ssim_value = ssim(im1,
                          im2,
                          multichannel=True,
                          win_size=7,
                          channel_axis=-1)

        all_frame_scores += ssim_value

    all_frame_scores /= len(keyframes)
    return {'ssim': all_frame_scores, 'state': 1}


# GenVideo clip
def compute_temporal_consistency(video_path):
    global model, preprocess

    all_frame_scores = 0.
    video_frames = load_video(video_path)[:16]  # get pre 16 frames
    for i in range(1, len(video_frames)):
        pre_frame = Image.fromarray(video_frames[i - 1])
        next_frame = Image.fromarray(video_frames[i])
        score = calculate_clip_score(model,
                                     preprocess,
                                     first_data=pre_frame,
                                     second_data=next_frame,
                                     first_flag='img',
                                     second_flag='img').cpu().numpy()
        all_frame_scores += score
    return all_frame_scores / (len(video_frames) - 1)


# GenVideo-Text clip
def compute_video_text_alignment(video_path, text):
    global model, preprocess

    all_frame_scores = 0.
    video_frames = load_video(video_path, )[:16]
    assert len(video_frames) != 0, 'check video_path {video_path}!'
    for frame in video_frames:
        frame = Image.fromarray(frame)
        score = calculate_clip_score(model,
                                     preprocess,
                                     first_data=text,
                                     second_data=frame,
                                     first_flag='txt',
                                     second_flag='img').cpu().numpy()
        all_frame_scores += score
    return all_frame_scores / len(video_frames)


# GenVideo-RefVideo clip (keyframes)
def compute_video_video_similarity(reference_video_path,
                                   video_path,
                                   keyframes=[0, 4, 8, 12, 15]):
    global model, preprocess

    reference_video = load_video(reference_video_path)
    try:
        video_frames = load_video(video_path,
                                  size=reference_video[0].shape[:2])
    except:
        print(video_path, ' Error !!')
        return {'clip': 0, 'state': 0}

    all_frame_scores = 0.
    for idx, frames in enumerate(keyframes):
        im1 = Image.fromarray(np.array(reference_video[10 + frames * 3]))
        im2 = Image.fromarray(np.array(video_frames[frames]))
        score = calculate_clip_score(model,
                                     preprocess,
                                     first_data=im1,
                                     second_data=im2,
                                     first_flag='img',
                                     second_flag='img').cpu().numpy()
        all_frame_scores += score

    all_frame_scores /= len(keyframes)
    return {'clip': all_frame_scores, 'state': 1}


# MSE (First) and SSIM (First)
def compute_image_image_similarity(init_image_path, video_path):
    global model, preprocess

    # all_frame_scores = 0.
    init_image = Image.fromarray(load_image(init_image_path))
    try:
        video_frames = load_video(video_path,
                                  size=init_image.size)  # size=(512, 904)
    except:
        print(video_path, ' Error !!')
        return {'MSE': 0, 'SSIM': 0, 'state': 0}
    init_image_np = np.array(init_image)
    video_frame_np = np.array(video_frames[0])  # or 10th frames

    if init_image_np.shape == video_frame_np.shape:
        mse_value = mean_squared_error(init_image_np, video_frame_np)
        ssim_value = ssim(init_image_np,
                          video_frame_np,
                          multichannel=True,
                          win_size=7,
                          channel_axis=-1)
    else:
        print("Error: The images do not have the same dimensions.", video_path)
        return {'MSE': 0, 'SSIM': 0, 'state': 0}

    return {'MSE': mse_value, 'SSIM': ssim_value, 'state': 1}


# Image-GenVideo clip
def compute_video_image_similarity(video_path, image_path):
    global model, preprocess

    all_frame_scores = 0.
    image = Image.fromarray(load_image(image_path))

    try:
        video_frames = load_video(video_path, size=image.size)
    except:
        print(video_path, ' Error !!')
        return {"clip_per": all_frame_scores / len(video_frames), "state": 0}

    for frame in video_frames:
        frame = Image.fromarray(frame)
        score = calculate_clip_score(model,
                                     preprocess,
                                     first_data=image,
                                     second_data=frame,
                                     first_flag='img',
                                     second_flag='img').cpu().numpy()
        all_frame_scores += score
    return {"clip_per": all_frame_scores / len(video_frames), "state": 1}
