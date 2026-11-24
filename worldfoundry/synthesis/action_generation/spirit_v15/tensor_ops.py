# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team. Modified by WorldFoundry for inference-only use.
# ==============================================================================

import torch
import torch.nn.functional as F


def pad_and_cat(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = F.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    stacked_tensor = torch.cat(padded_tensors, dim=1)
    return stacked_tensor
