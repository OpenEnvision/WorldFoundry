"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
from PIL import Image
from tqdm import tqdm
from typing import *
import clip
import os
import json
import re

try:
    from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters  # noqa: F401
except ImportError:
    fleiss_kappa = aggregate_raters = None

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from video_dataset import *
from metrics import *
import argparse
import time
import datetime

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def escape(x):
    return x.replace('-', '_').replace('/', '_')


def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


def get_clip(eval_model: Module, device: Union[torch.device, int]) \
        -> Tuple[Module, Module]:
    """Get the CLIP model

    Args:
        eval_model (Module): The CLIP model to evaluate
        device (Union[torch.device, int]): Device index to select

    Returns:
        Tuple[Module, Module]: The CLIP model and a preprocessor
    """
    clip_model, _ = clip.load(eval_model)
    clip_model = clip_model.cuda(device)
    clip_prep = T.Compose([T.Resize(224),
                           T.CenterCrop(224),
                           T.ToTensor(),
                           T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))])
    return clip_model, clip_prep


def init_metric(root: str, metric: Type[Metric], eval_model: Module,
                limit: int, device: torch.device) -> Metric:
    """Initialize a given metric class.

    Args:
        root (str): Path to data directory
        metric (Type[Metric]): Metric class
        eval_model (Module): Evaluating CLIP model
        limit (int, optional): Number of reference samples
        device (torch.device): Device index to select

    Returns:
        Metric: Initialized metric instance
    """
    m = metric(768 if eval_model == 'ViT-L/14' else 512,
                limit=limit)
    m.cuda(device)
    m._debug = False
    return m


@torch.no_grad()
def populate_metrics(dataloader: DataLoader, metrics: List[Metric],
                     clip_model: Module,) -> Tensor:
    """Populate the list of metrics using a given data loader.

    Args:
        dataloader (DataLoader): Data loader
        metrics (List[Metric]): List of metrics
        clip_model (Module): Evaluating CLIP model
        num_frames: the number of frame for each generated video

    Returns:
        Tensor: Labels
    """
    device = next(clip_model.parameters()).device
    for i, (fake, caption, vid) in enumerate(
            tqdm(dataloader)):

        fake = fake.cuda(device)    # [batch_size, frame_num, 3, h, w]
        batch_size = fake.shape[0]
        fake_frame_num = fake.shape[1]

        fake = fake.reshape(-1, fake.shape[-3], fake.shape[-2], fake.shape[-1])    # [batch_size * frame_num, 3, h, w]

        txt = clip.tokenize(caption, truncate=True).cuda(device)
        txt_features = clip_model.encode_text(txt).float()

        # the frames have already been processed by clip_prep in the video_dataset.py
        fake_im_features = clip_model.encode_image(
            fake).float()                      # [batch_size * frame_num, hidden_size]

        # float16 of CLIP may suffer in l2-normalization
        txt_features = F.normalize(txt_features, dim=-1)
        fake_im_features = F.normalize(fake_im_features, dim=-1)

        Y_ref = txt_features
        Y_ref = txt_features
        X = fake_im_features.reshape(batch_size, fake_frame_num, fake_im_features.shape[-1])        # [batch_size, frame_num, hidden_size]

        # metrics handle features in float64
        for idx, m in enumerate(metrics):
            m.update(None, Y_ref, F.normalize(X.mean(1), dim=-1))

        if (i + 1) * Y_ref.shape[0] > metrics[0].limit:
            print(f"break loop due to the limit of {metrics[0].limit}.")
            break

def blipscore(config_path, args):
    import ruamel.yaml as yaml
    from worldfoundry.base_models.perception_core.video_text.fetv_blip import utils
    from worldfoundry.base_models.perception_core.video_text.fetv_blip.models.blip_retrieval import (
        blip_retrieval,
    )

    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    #### Dataset #### 
    print("Creating BLIP dataset")
    test_dataset = VideoDataset(gen_path=args.gen_path, prompt_file=args.prompt_file, 
                                media_size=config['image_size'], max_num_frm=config['num_frm_test'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                    drop_last=False, shuffle=False,
                    num_workers=8)

    #### Model #### 
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'])
    model = model.to(args.device)
    model_without_ddp = model

    return compute_blipscore(model_without_ddp, test_loader, model_without_ddp.tokenizer, args.device, config, utils)

@torch.no_grad()
def compute_blipscore(model, data_loader, tokenizer, device, config, utils, use_itm=True):
    """
        use_itm: whether to use the image-text matching module. If not, the score is computed as the dot-product between visual and textual encoder outputs.
    """
    # test
    model.eval() 
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = list(data_loader.dataset.captions.values())
    texts = [pre_caption(t, 40) for t in texts]
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    print('Computing text features for evaluation...')
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = tokenizer.additional_special_tokens_ids[0]
    
    print('Computing video features for evaluation...')
    video_feats = []
    video_embeds = []
    for i, (video, _, _) in enumerate(tqdm(data_loader)): 

        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H)
        video = video.to(device,non_blocking=True) 
        video_feat = model.visual_encoder(video)        
        video_embed = model.vision_proj(video_feat[:,0,:])   
        video_embed = video_embed.view(B,N,-1).mean(dim=1)
        video_embed = F.normalize(video_embed,dim=-1)  
       
        video_feat = video_feat.view(B,-1,video_feat.shape[-1])
        video_feats.append(video_feat.cpu())
        video_embeds.append(video_embed)
     
    video_feats = torch.cat(video_feats,dim=0)
    video_embeds = torch.cat(video_embeds,dim=0)

    def dot(x, y):
        return (x * y).sum(dim=-1)
    dot_sims = dot(video_embeds, text_embeds)

    if use_itm:
        print('Computing itm scores...')
        start_time = time.time()
        itm_scores = []
        for i in tqdm(range(0, num_text, config['batch_size'])):
            encoder_output = video_feats[i: min(num_text, i+config['batch_size'])].to(device,non_blocking=True) 
            encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device,non_blocking=True)
            output = model.text_encoder(text_ids[i: min(num_text, i+config['batch_size'])], 
                                        attention_mask = text_atts[i: min(num_text, i+config['batch_size'])],
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,                             
                                        return_dict = True,
                                    )
            score = model.itm_head(output.last_hidden_state[:,0,:])
            itm_scores.append(score)
        itm_scores = torch.cat(itm_scores, dim=0)     
        itm_scores = F.softmax(itm_scores, dim=1)[:,1]
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str)) 

    return dot_sims, itm_scores

if "__main__" == __name__:
    # config
    parser = argparse.ArgumentParser()     
    parser.add_argument('--eval_model', default=None, help="CLIP model checkpoint")
    parser.add_argument('--is_clip_ft', default=False, type=str2bool, help="Whether the CLIP model is fine-tuned or not, set to True when computing CLIPScore-ft")
    parser.add_argument('--blip_config', default=None)
    parser.add_argument('--prompt_file', default='fetv_data.json')        
    parser.add_argument('--save_results', default=True, type=str2bool)
    parser.add_argument('--result_path', default='auto_eval_results', help='path to save the evaluation results.')        
    parser.add_argument('--gen_path', default=None, help='path to the generated videos, processed into frames')      
    # The value only names the output file.  Keeping the upstream paper-model
    # choices here prevents the official metric from evaluating a new model.
    parser.add_argument('--t2v_model', required=True, help="Name of the text2video generation model")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_frm_num', default=64, type=int)
    parser.add_argument('--limit', default=1000, type=int, help='number of sampels to evaluate')
    args = parser.parse_args()

    _ = torch.manual_seed(args.seed)
    if args.eval_model is None:
        args.eval_model = "ViT-B/32"

    METRICS = [ClipScore]

    # get clip model
    args.device = torch.device("cuda:0")

    print(f"Loading {args.eval_model}...")
    clip_model, clip_prep = get_clip(args.eval_model, args.device)

    metrics = [
        init_metric("datas", x, args.eval_model, args.limit, args.device) for x in METRICS]

    # load dataset
    ds = VideoDataset(gen_path=args.gen_path, prompt_file=args.prompt_file, 
                      max_num_frm=args.max_frm_num, preprocessor=clip_prep)
    dl = DataLoader(ds, batch_size=16,
                    drop_last=False, shuffle=False,
                    num_workers=8)


    # compute clip features
    populate_metrics(dl, metrics, clip_model)
    clip_model.to('cpu')
    results = {m.name: m.compute(reduction=False) for m in metrics}


    # remove the unusual prompt ids which do not have real videos
    remove_ids = list(range(531,609)) if 'real_videos' in args.gen_path else []

    # compute metrics using blip
    if args.blip_config is not None:
        dot_sims, _ = blipscore(args.blip_config, args)
        results.update({'BLIPScore': dot_sims})

    # TorchMetrics may squeeze a one-sample state to a scalar.  Preserve the
    # official sent-index mapping for bounded validation instead of failing on
    # iteration over a 0-d tensor.
    results_per_sent_id = {
        metric: {
            i: score.cpu().item()
            for i, score in enumerate(torch.atleast_1d(result))
            if i not in remove_ids
        }
        for metric, result in results.items()
    }

    if args.save_results:
        for metric in results_per_sent_id:
            suffix = '-ft' if args.is_clip_ft and metric=='CLIPScore' else ''
            result_path = os.path.join(args.result_path, metric+suffix)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            result_file = os.path.join(result_path, f"auto_eval_results_{args.t2v_model}.json")
            with open(result_file, 'w') as f:
                json.dump(results_per_sent_id[metric], f)
