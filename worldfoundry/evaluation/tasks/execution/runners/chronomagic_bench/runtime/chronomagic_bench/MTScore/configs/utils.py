import numpy as np
import cv2
import os
import torch
from worldfoundry.base_models.llm_mllm_core.mllm.internvideo2.multi_modality.models import (
    InternVideo2_Stage2,
)
from worldfoundry.base_models.llm_mllm_core.mllm.umt.models.backbones.bert.tokenization_bert import (
    BertTokenizer,
)
from worldfoundry.base_models.llm_mllm_core.mllm.internvideo2.multi_modality.models.backbones.internvideo2.pos_embed import (
    interpolate_pos_embed_internvideo2_new,
)


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
        
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def get_text_feat_dict(texts, clip, text_feat_d={}):
    for t in texts:
        feat = clip.get_txt_feat(t)
        text_feat_d[t] = feat
    return text_feat_d


def get_vid_feat(frames, vlm):
    return vlm.get_vid_features(frames)


def retrieve_text(frames, 
                  texts, 
                  model,
                  topk:int=5,
                  config: dict={},
                  device=torch.device('cuda')):
    
    vlm = model
    vlm = vlm.to(device)
    
    fn = config.get('num_frames', 8)
    size_t = config.get('size_t', 224)
    frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
    vid_feat = vlm.get_vid_feat(frames_tensor)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, vlm, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    probs, idxs = vlm.predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.long().numpy()[0].tolist()]
    return ret_texts, probs.float().numpy()[0]


def setup_internvideo2(config: dict):
    if "bert" in config.model.text_encoder.name:
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained, local_files_only=False)
        model = InternVideo2_Stage2(config=config, tokenizer=tokenizer, is_pretrain=True)
    else:
        model = InternVideo2_Stage2(config=config, is_pretrain=True)
        tokenizer = model.tokenizer

    if config.get('compile_model', False):
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    model = model.to(torch.device(config.device))
    model_without_ddp = model

    if (config.pretrained_path.strip() and (os.path.isfile(config.pretrained_path)) or "s3://" in config.pretrained_path):
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        try:
            if "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint["module"] # This is a deepspeed stage 1 model
        except:  
            state_dict = checkpoint

        if config.get('origin_num_frames', None) is not None:
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(state_dict, model_without_ddp.vision_encoder, orig_t_size=config.origin_num_frames)
            assert a == len(state_dict), state_dict.keys()

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(f"load_state_dict: {msg}")
    
    if config.get('use_bf16', False):
        model_without_ddp = model_without_ddp.to(torch.bfloat16)
    elif config.get('use_half_precision', False):
        model_without_ddp = model_without_ddp.to(torch.float16)
    else:
        model_without_ddp = model_without_ddp.to(torch.float32)
    model_without_ddp.eval()
    return (model_without_ddp, tokenizer,)

