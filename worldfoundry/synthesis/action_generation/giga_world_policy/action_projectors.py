import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbodimentSpecificLinear(nn.Module):
    """Linear layer with per-embodiment weights and biases."""

    def __init__(self, input_dim: int, output_dim: int, num_categories: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_categories = int(num_categories)
        self.weight = nn.Parameter(torch.empty(self.num_categories, input_dim, output_dim, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(self.num_categories, output_dim, dtype=dtype))

        k = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.weight, -k, k)
        nn.init.uniform_(self.bias, -k, k)

    def _normalize_embodiment_id(self, embodiment_id: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if embodiment_id is None:
            embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=self.weight.device)
        elif not isinstance(embodiment_id, torch.Tensor):
            embodiment_id = torch.as_tensor(embodiment_id, dtype=torch.long, device=self.weight.device)
        else:
            embodiment_id = embodiment_id.to(device=self.weight.device)

        if embodiment_id.dtype != torch.long:
            embodiment_id = embodiment_id.to(torch.long)
        embodiment_id = embodiment_id.reshape(-1)
        if embodiment_id.numel() == 1 and batch_size != 1:
            embodiment_id = embodiment_id.expand(batch_size)
        if embodiment_id.numel() != batch_size:
            raise ValueError(
                f"embodiment_id batch size mismatch: got {tuple(embodiment_id.shape)}, expected batch={batch_size}"
            )
        if torch.any((embodiment_id < 0) | (embodiment_id >= self.num_categories)):
            raise ValueError(
                f"embodiment_id out of range for num_categories={self.num_categories}: {embodiment_id.detach().cpu().tolist()}"
            )
        return embodiment_id

    def forward(self, x: torch.Tensor, embodiment_id: torch.Tensor | None = None) -> torch.Tensor:
        embodiment_id = self._normalize_embodiment_id(embodiment_id, batch_size=x.shape[0])
        one_hot = F.one_hot(embodiment_id, num_classes=self.num_categories).to(dtype=self.weight.dtype)
        selected_weight = torch.einsum("bc,cij->bij", one_hot, self.weight)
        selected_bias = torch.einsum("bc,cj->bj", one_hot, self.bias)
        out = torch.bmm(x.to(dtype=selected_weight.dtype), selected_weight) + selected_bias.unsqueeze(1)
        return out.to(dtype=x.dtype)

    def extra_repr(self):
        return (
            f"num_categories={self.num_categories}, input_dim={self.weight.shape[1]}, output_dim={self.weight.shape[2]}"
        )


class EmbodimentSpecificActionEncoder(nn.Module):
    uses_embodiment_id = True

    def __init__(self, action_dim: int, inner_dim: int, num_embodiments: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_proj = EmbodimentSpecificLinear(action_dim, 128, num_categories=num_embodiments, dtype=dtype)
        self.act1 = nn.GELU()
        self.mid_proj = nn.Linear(128, 256)
        self.act2 = nn.GELU()
        self.out_proj = nn.Linear(256, inner_dim)

    def forward(self, action: torch.Tensor, embodiment_id: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_proj(action, embodiment_id=embodiment_id)
        x = self.act1(x)
        x = self.mid_proj(x)
        x = self.act2(x)
        return self.out_proj(x)


class EmbodimentSpecificActionDecoder(nn.Module):
    uses_embodiment_id = True

    def __init__(self, inner_dim: int, action_dim: int, num_embodiments: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_proj = nn.Linear(inner_dim, 256)
        self.act1 = nn.GELU()
        self.mid_proj = nn.Linear(256, 128)
        self.act2 = nn.GELU()
        self.out_proj = EmbodimentSpecificLinear(128, action_dim, num_categories=num_embodiments, dtype=dtype)

    def forward(self, action_states: torch.Tensor, embodiment_id: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_proj(action_states)
        x = self.act1(x)
        x = self.mid_proj(x)
        x = self.act2(x)
        return self.out_proj(x, embodiment_id=embodiment_id)
