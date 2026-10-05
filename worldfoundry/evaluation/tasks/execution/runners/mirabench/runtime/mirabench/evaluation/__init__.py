import os

import torch
import numpy as np
import imageio.v3 as iio
from easydict import EasyDict as edict
from evaluation.consistency_3D import EvaluateErrBetweenTwoImage
from evaluation.temporal_dino_consistency import EvaluateTemporalDinoConsistency
from evaluation.temporal_clip_consistency import EvaluateTemporalClipConsistency
from evaluation.motion_smoothness import EvaluateMotionSmoothness,MotionSmoothness
from evaluation.dynamic_degree import EvaluateDynamicDegree,DynamicDegree
from evaluation.aesthetic_quality import get_aesthetic_model,EvaluateLaionAesthetic
from evaluation.imaging_quality import MUSIQ,EvaluateImagingQuality
from evaluation.text_video_consistency import (
    EvaluateTextVideoConsistency,
    SimpleTokenizer,
    ViCLIP,
    get_video_feature,
)
from worldfoundry.base_models.perception_core.frame_interpolation.amt import (
    checkpoint_path as amt_checkpoint_path,
    config_path as amt_config_path,
)
from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from worldfoundry.base_models.perception_core.video_text.viclip import checkpoint_path as viclip_checkpoint_path
from worldfoundry.base_models.perception_core.optical_flow.raft import checkpoint_path as raft_checkpoint_path
from worldfoundry.base_models.perception_core.general_perception.dino_embeddings import load_dino_vitb16_feature_model
from worldfoundry.base_models.perception_core.tracking.cotracker import CoTrackerOnlinePredictor
from worldfoundry.base_models.capabilities import vbench_asset_path
from worldfoundry.core.io import list_numbered_frame_paths

class metrics_calculator():
    def __init__(self,metrics,ckpt_path="data/ckpt",device="cuda"):
        print(f"Initializing metrics: {metrics}")
        self.ckpt_path=ckpt_path
        self.device=device
        self._text_video_feature_path = None
        self._text_video_feature = None
        if "temporal_dino_consistency" in metrics:
            self.temporal_dino_consistency_dino_model = load_dino_vitb16_feature_model(device=self.device)
        if "temporal_clip_consistency" in metrics:
            self.temporal_clip_consistency_clip_model, self.temporal_clip_consistency_preprocess = clip.load("ViT-B/32", device=self.device)
        if "temporal_motion_smoothness" in metrics:
            temporal_motion_smoothness_motion_model_config_path=str(amt_config_path())
            temporal_motion_smoothness_motion_model_ckpt=str(amt_checkpoint_path())
            self.temporal_motion_smoothness_motion_model = MotionSmoothness(temporal_motion_smoothness_motion_model_config_path, temporal_motion_smoothness_motion_model_ckpt, self.device)
        if "dynamic_degree" in metrics:
            dynamic_degree_model_ckpt=str(raft_checkpoint_path())
            self.dynamic_degree_frame_interval=1
            self.dynamic_degree_model=DynamicDegree(edict({"model":dynamic_degree_model_ckpt, "small":False, "mixed_precision":False, "alternate_corr":False}),device=self.device)
        if "tracking_strength" in metrics:
            self.tracking_strength_model_cotracker = CoTrackerOnlinePredictor().to(self.device)
            self.tracking_strength_grid_size=10
            self.tracking_strength_frame_interval=1
        if set(['3D_consistency_num_pts','3D_consistency_num_inliers_F','3D_consistency_keep_ratio','3D_consistency_mean_err','3D_consistency_rmse'])&set(metrics):
            self.consistency_3D_interval_list=[60, 50, 40, 30, 20, 10] # default setting
            # self.consistency_3D_interval_list=[20, 10] # default setting
            self.consistency_3D_ransac_th=3
            self._three_d_pair_cache = {}
            self._three_d_folder_cache = {}
        if "aesthetic_quality" in metrics:
            self.aesthetic_quality_model=get_aesthetic_model(ckpt_path).to(self.device)
            self.aesthetic_quality_clip_model, self.aesthetic_quality_preprocess = clip.load('ViT-L/14', device=self.device)
        if "imaging_quality" in metrics:
            imaging_quality_model_ckpt=str(vbench_asset_path("vbench_musiq_spaq_checkpoint"))
            self.imaging_quality_model=MUSIQ(pretrained_model_path=imaging_quality_model_ckpt)
            self.imaging_quality_model.to(self.device)
            self.imaging_quality_model.training = False
        if set(['camera_alignment','main_object_alignment','background_alignment','style_alignment','overall_consistency'])&set(metrics):
            text_video_consistency_model_viclip_ckpt=str(viclip_checkpoint_path())
            text_video_consistency_model_viclip_tokenizerp_ckpt = None

            self.text_video_consistency_model_viclip_tokenizer=SimpleTokenizer(text_video_consistency_model_viclip_tokenizerp_ckpt)
            self.text_video_consistency_model_viclip = ViCLIP(tokenizer= self.text_video_consistency_model_viclip_tokenizer, pretrain=text_video_consistency_model_viclip_ckpt).to(self.device)


    # temporal consistency
    def calculate_temporal_dino_consistency(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateTemporalDinoConsistency(self.temporal_dino_consistency_dino_model,store_image_folder,self.device)

    def calculate_temporal_clip_consistency(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateTemporalClipConsistency(self.temporal_clip_consistency_clip_model,self.temporal_clip_consistency_preprocess,store_image_folder,self.device)

    def calculate_temporal_motion_smoothness(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateMotionSmoothness(self.temporal_motion_smoothness_motion_model,store_image_folder,self.device)

    # temporal motion strength
    def calculate_dynamic_degree(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateDynamicDegree(self.dynamic_degree_model,store_image_folder,self.dynamic_degree_frame_interval)

    def calculate_tracking_strength(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        imgs=list_numbered_frame_paths(store_image_folder)
        if not imgs:
            raise ValueError("tracking strength requires at least one frame")
        frame_array = np.stack(
            [iio.imread(img) for img in imgs[::self.tracking_strength_frame_interval]],
            axis=0,
        )
        video = torch.from_numpy(frame_array).permute(0, 3, 1, 2).unsqueeze(0).float()
        if torch.device(self.device).type == "cuda":
            video = video.pin_memory()
        video = video.to(self.device, non_blocking=True)  # B T C H W
        all_pred_tracks=[]
        with torch.inference_mode():
            self.tracking_strength_model_cotracker(
                video_chunk=video,
                is_first_step=True,
                grid_size=self.tracking_strength_grid_size,
            )
            for ind in range(
                0,
                video.shape[1] - self.tracking_strength_model_cotracker.step,
                self.tracking_strength_model_cotracker.step,
            ):
                pred_tracks, _ = self.tracking_strength_model_cotracker(
                    video_chunk=video[:, ind : ind + self.tracking_strength_model_cotracker.step * 2]
                )
                all_pred_tracks.append(pred_tracks[0].detach())

        if not all_pred_tracks:
            raise ValueError("tracking strength video is shorter than the CoTracker window")
        tracks = torch.cat(all_pred_tracks, dim=0)
        displacement = tracks - tracks[0]
        return float(torch.linalg.vector_norm(displacement, dim=-1).mean().item())

    # 3D consistency
    def _evaluate_3d_pair(self, left_img_path, right_img_path, ransac_threshold):
        cache_key = (left_img_path, right_img_path, float(ransac_threshold))
        if cache_key not in self._three_d_pair_cache:
            if len(self._three_d_pair_cache) >= 1024:
                self._three_d_pair_cache.clear()
            self._three_d_pair_cache[cache_key] = EvaluateErrBetweenTwoImage(
                left_img_path,
                right_img_path,
                ransac_threshold,
            )
        return self._three_d_pair_cache[cache_key]

    def _calculate_3d_metrics(self, store_image_folder):
        cache_key = os.path.abspath(str(store_image_folder))
        cached = self._three_d_folder_cache.get(cache_key)
        if cached is not None:
            return cached
        frame_paths = list_numbered_frame_paths(store_image_folder)

        totals = {
            "3D_consistency_num_pts": 0.0,
            "3D_consistency_num_inliers_F": 0.0,
            "3D_consistency_keep_ratio": 0.0,
            "3D_consistency_mean_err": 0.0,
            "3D_consistency_rmse": 0.0,
        }
        pair_count = 0
        end_frame = len(frame_paths)
        for interval_num in self.consistency_3D_interval_list:
            match_inter = int((end_frame - interval_num) / 5)
            if match_inter <= 0:
                print(f"can not match at interval={interval_num}")
                continue
            for left_id in range(0, end_frame, match_inter):
                right_id = left_id + interval_num
                if right_id > end_frame - 1:
                    break
                mean_error, _, rmse, _, keep_rate, num_inliers, num_points = self._evaluate_3d_pair(
                    str(frame_paths[left_id]),
                    str(frame_paths[right_id]),
                    self.consistency_3D_ransac_th,
                )
                totals["3D_consistency_num_pts"] += float(num_points)
                totals["3D_consistency_num_inliers_F"] += float(num_inliers)
                totals["3D_consistency_keep_ratio"] += float(keep_rate)
                totals["3D_consistency_mean_err"] += float(mean_error)
                totals["3D_consistency_rmse"] += float(rmse)
                pair_count += 1

        result = {
            metric: total / pair_count if pair_count else float("nan")
            for metric, total in totals.items()
        }
        if len(self._three_d_folder_cache) >= 256:
            self._three_d_folder_cache.clear()
        self._three_d_folder_cache[cache_key] = result
        return result

    def calculate_3D_consistency_num_pts(self, store_image_folder, *_):
        return self._calculate_3d_metrics(store_image_folder)["3D_consistency_num_pts"]

    def calculate_3D_consistency_num_inliers_F(self, store_image_folder, *_):
        return self._calculate_3d_metrics(store_image_folder)["3D_consistency_num_inliers_F"]

    def calculate_3D_consistency_keep_ratio(self, store_image_folder, *_):
        return self._calculate_3d_metrics(store_image_folder)["3D_consistency_keep_ratio"]

    def calculate_3D_consistency_mean_err(self, store_image_folder, *_):
        return self._calculate_3d_metrics(store_image_folder)["3D_consistency_mean_err"]

    def calculate_3D_consistency_rmse(self, store_image_folder, *_):
        return self._calculate_3d_metrics(store_image_folder)["3D_consistency_rmse"]

    # video frame quality
    def calculate_aesthetic_quality(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateLaionAesthetic(self.aesthetic_quality_model,self.aesthetic_quality_clip_model,self.aesthetic_quality_preprocess,store_image_folder,self.device)

    def calculate_imaging_quality(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateImagingQuality(self.imaging_quality_model,store_image_folder,self.device)

    # text-video alignment
    def _calculate_text_video_consistency(self, video_path, text):
        video_key = os.path.abspath(str(video_path))
        if self._text_video_feature_path != video_key:
            self._text_video_feature = get_video_feature(
                self.text_video_consistency_model_viclip,
                video_path,
                self.device,
            )
            self._text_video_feature_path = video_key
        return EvaluateTextVideoConsistency(
            self.text_video_consistency_model_viclip,
            video_path,
            self.text_video_consistency_model_viclip_tokenizer,
            self.device,
            text,
            video_feature=self._text_video_feature,
        )

    def calculate_camera_alignment(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if camera_caption is not None:
            return self._calculate_text_video_consistency(video_path, camera_caption)
        else:
            return None

    def calculate_main_object_alignment(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if main_object_caption is not None:
            return self._calculate_text_video_consistency(video_path, main_object_caption)
        else:
            return None

    def calculate_background_alignment(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if background_caption is not None:
            return self._calculate_text_video_consistency(video_path, background_caption)
        else:
            return None
    
    def calculate_style_alignment(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if style_caption is not None:
            return self._calculate_text_video_consistency(video_path, style_caption)
        else:
            return None

    def calculate_overall_consistency(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if dense_caption is not None:
            return self._calculate_text_video_consistency(video_path, dense_caption)
        elif short_caption is not None:
            return self._calculate_text_video_consistency(video_path, short_caption)
        else:
            return None


    def __call__(self,metric,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        calculator = getattr(self, f"calculate_{metric}", None)
        if not callable(calculator):
            raise ValueError(f"unsupported MiraBench metric: {metric}")
        return calculator(
            store_image_folder,
            video_path,
            short_caption,
            dense_caption,
            main_object_caption,
            background_caption,
            style_caption,
            camera_caption,
        )
        
