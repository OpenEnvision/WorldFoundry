import os

import cv2
import torch

from worldfoundry.base_models.perception_core.optical_flow.raft import InputPadder, RAFT
from worldfoundry.core.io import list_numbered_frame_paths

class DynamicDegree:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.load_model()
    

    def load_model(self):
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.model))

        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()



    def get_score(self, img, flo):
        del img
        magnitude = torch.linalg.vector_norm(flo[0].float(), dim=0).flatten()
        cut_index = int(magnitude.numel() * 0.05)
        if cut_index < 1:
            return magnitude.new_tensor(float("nan"))
        return magnitude.topk(cut_index, sorted=False).values.mean()


    # def set_params(self, frame, count):
    #     scale = min(list(frame.shape)[-2:])
    #     self.params = {"thres":6.0*(scale/256.0), "count_num":round(4*(count/16.0))}


    def infer(self, video_path,frame_interval):
        with torch.inference_mode():
            if video_path.endswith('.mp4'):
                frames = self.get_frames(video_path)
            elif os.path.isdir(video_path):
                frames = self.get_frames_from_img_folder(video_path)
            else:
                raise NotImplementedError
            frames=frames[::frame_interval]
            if len(frames) < 2:
                return float("nan")
            frame_batch = torch.stack(frames, dim=0)
            if torch.device(self.device).type == "cuda":
                frame_batch = frame_batch.pin_memory()
            frame_batch = frame_batch.to(self.device, non_blocking=True)
            # self.set_params(frame=frames[0], count=len(frames))
            static_score = []
            for image1, image2 in zip(frame_batch[:-1], frame_batch[1:]):
                image1 = image1.unsqueeze(0)
                image2 = image2.unsqueeze(0)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                static_score.append(max_rad)
            # whether_move = self.check_move(static_score)
            if not static_score:
                return float("nan")
            return float(torch.stack(static_score).mean().item())

    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) # get fps
        interval = max(round(fps/8), 1)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        frame_list = self.extract_frame(frame_list, interval)
        return frame_list 
    
    
    def extract_frame(self, frame_list, interval=1):
        extract = []
        for i in range(0, len(frame_list), interval):
            extract.append(frame_list[i])
        return extract


    def get_frames_from_img_folder(self, img_folder):
        frame_list = []
        imgs = list_numbered_frame_paths(img_folder)
        for img in imgs:
            frame = cv2.imread(str(img), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"failed to read frame: {img}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()
            frame_list.append(frame)
        assert frame_list != []
        return frame_list



def EvaluateDynamicDegree(dynamic, store_image_folder,dynamic_degree_frame_interval):
    with torch.inference_mode():
        return dynamic.infer(store_image_folder,dynamic_degree_frame_interval)
