import torch
from timm.models.vision_transformer import Mlp
from torch import nn

from .configuration_action_expert import ActionExpertConfig
from .modeling_action_expert import ActionExpertModel, ActionExpertPretrainedModel
from .modeling_internlm2 import InternLM2RMSNorm


class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = InternLM2RMSNorm(hidden_size)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class LatentPlannerModel(ActionExpertPretrainedModel):
    """Action Expert model based on InternLM2, the main difference is as follows:
    - hidden dim and intermediate dim shrinks to the half of original InternLM2-2B
    - text embedding layer replaced by action projection layers, converting action into hiddent states
    - initial action token initialized as noisy action tokens instead of input tokens
    - each docoder layer accepts kv states from corresponding LLM layer, and used as condition token for action
      generation
    - add new linear projector to squeeze kv states dim to align with action hidden states dim
    - causal mask for VLM tokens `I`, robotic states `q` and action tokens `a`
    """

    def __init__(
        self,
        action_head_dim: int,
        real_action_config: ActionExpertConfig,
        latent_planner_config: ActionExpertConfig,
        torch_dtype=torch.bfloat16,
        latent_token_nums: int = 4,
    ):
        super().__init__(real_action_config)
        self.torch_dtype = torch_dtype

        self.latent_action_expert = ActionExpertModel._from_config(latent_planner_config, torch_dtype=self.torch_dtype)

        latent_action_hidden_size = latent_planner_config.hidden_size
        self.latent_action_dim = latent_planner_config.action_dim
        self.latent_action_head_dim = latent_action_hidden_size // latent_planner_config.num_attention_heads

        self.latent_k_proj_layers = nn.ModuleList(
            [
                nn.Linear(self.latent_action_head_dim, action_head_dim, dtype=self.torch_dtype)
                for _ in range(real_action_config.llm_config.num_hidden_layers)
            ]
        )
        self.latent_v_proj_layers = nn.ModuleList(
            [
                nn.Linear(self.latent_action_head_dim, action_head_dim, dtype=self.torch_dtype)
                for _ in range(real_action_config.llm_config.num_hidden_layers)
            ]
        )

        self.latent_num_labels = latent_planner_config.vocab_size
        self.latent_token_nums = latent_token_nums
        self.score = nn.Linear(latent_action_hidden_size, self.latent_num_labels, bias=False)
        self.latent_final_layer = FinalLayer(latent_action_hidden_size, self.latent_action_dim).to(self.torch_dtype)
        self.latent_state_action_adaptor = nn.Sequential(  # 1 -- > 1024 / 128 -- > 1024
            nn.Linear(latent_planner_config.action_dim, latent_planner_config.hidden_size, dtype=self.torch_dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(latent_planner_config.hidden_size, latent_planner_config.hidden_size, dtype=self.torch_dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(latent_planner_config.hidden_size, latent_planner_config.hidden_size, dtype=self.torch_dtype),
        )

    def latent_action_process(
        self,
        state: torch.Tensor = None,
        vlm_key_values_downsample: list | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        H = self.latent_token_nums
        action = torch.cat([state] * H, dim=1)  # if continuous [B, 4, 1]
        state_action_trajs = self.latent_state_action_adaptor(action)  # torch.Size([B, 4, 1024])
        outputs = self.latent_action_expert(state_action_trajs, attention_mask, vlm_key_values_downsample)
        vlm_key_values = outputs.past_key_values  # kv_cache(multi_head) of each decoder layer why there a layer of 25

        # Project vlm kv_cache head dim into action expert head dim
        latent_vlm_key_values_downsample = []
        for vlm_key_value, k_proj, v_proj in zip(
            vlm_key_values, self.latent_k_proj_layers, self.latent_v_proj_layers, strict=False
        ):
            latent_vlm_key_values_downsample.append((k_proj(vlm_key_value[0]), v_proj(vlm_key_value[1])))
        return outputs, latent_vlm_key_values_downsample

    def forward(
        self,
        vlm_key_values_downsample: tuple[tuple[torch.FloatTensor]] | None = None,
        attention_mask: torch.Tensor = None,
    ):
        B = attention_mask.shape[0]
        dtype = self.latent_state_action_adaptor[0].weight.dtype
        latent_state = torch.full((B, 1, 1), -1, dtype=dtype, device=attention_mask.device)

        outputs_latent, latent_vlm_key_values_downsample = self.latent_action_process(
            latent_state, vlm_key_values_downsample, attention_mask
        )

        return (
            latent_vlm_key_values_downsample,
            outputs_latent,
        )
