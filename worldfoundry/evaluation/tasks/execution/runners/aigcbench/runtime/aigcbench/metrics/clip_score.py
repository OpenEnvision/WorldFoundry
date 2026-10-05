"""
Calculates the CLIP Scores
"""
from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
import torch


def forward_modality(model, preprocess, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        data = preprocess(data).unsqueeze(0)
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        data = clip.tokenize(data)
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    # print(flag, features.shape)
    return features


@torch.no_grad()
def calculate_clip_score(model,
                         preprocess,
                         first_data,
                         second_data,
                         first_flag='txt',
                         second_flag='img'):
    # logit_scale = model.logit_scale.exp()
    first_features = forward_modality(model, preprocess, first_data,
                                      first_flag)
    second_features = forward_modality(model, preprocess, second_data,
                                       second_flag)

    # normalize features
    first_features = first_features / first_features.norm(
        dim=1, keepdim=True).to(torch.float32)
    second_features = second_features / second_features.norm(
        dim=1, keepdim=True).to(torch.float32)

    # calculate scores
    # score = logit_scale * (second_features * first_features).sum()
    score = (second_features * first_features).sum()
    return score
