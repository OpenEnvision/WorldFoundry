import torch


class VanillaCFG:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        x_pred = x_u + self.scale * (x_c - x_u)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = {}
        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat([uc[k], c[k]], 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class IdentityGuider:
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = {}
        for k in c:
            c_out[k] = c[k]
        return x, s, c_out
