"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from math import log2, pi

import torch


def generate_fourier_features(
    x: torch.Tensor,
    dim: int = 512,
    max_freq: int = 64,
    use_cos: bool = False,
    use_log: bool = False,
    cat_orig: bool = False,
):
    x_orig = x
    device, dtype, input_dim = x.device, x.dtype, x.shape[-1]
    num_bands = dim // (2 * input_dim) if use_cos else dim // input_dim

    if use_log:
        scales = 2.0 ** torch.linspace(0.0, log2(max_freq), steps=num_bands, device=device, dtype=dtype)
    else:
        scales = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)

    x = x.unsqueeze(-1)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat(
        (
            [x.sin(), x.cos()]
            if use_cos
            else [
                x.sin(),
            ]
        ),
        dim=-1,
    )
    x = x.flatten(-2)
    if cat_orig:
        return torch.cat((x, x_orig), dim=-1)
    return x


# from PIL import Image
# from unidepth.utils import image_grid, colorize
# if __name__ == "__main__":
#     H, W = 512, 512
#     resolution = 128
#     mesh = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
#     mesh = torch.stack(mesh, dim=0).unsqueeze(0)
#     mesh = mesh.view(1, 2, -1).permute(0, 2, 1)

#     features = generate_fourier_features(mesh, dim=32, max_freq=resolution, use_log=True)
#     channels = features.shape[-1]
#     print(features.shape)

#     features = features[0].view(H, W, channels).permute(2, 0, 1).numpy()
#     Image.fromarray(image_grid([colorize(1+x, 0.0, 2.0, "viridis") for x in features], rows=8, cols=4)).save(f"tmp_{resolution}.png")
