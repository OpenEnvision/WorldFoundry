import torch


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp
