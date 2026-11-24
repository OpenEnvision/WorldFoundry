# LaCT

import collections
import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])


def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y


def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    # TODO: log the update loss
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile
# TODO: add a version that uses the torch.compile
def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
    momentum: torch.Tensor | None = None,
    ttt_update_steps: int = 1,
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]

    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    if momentum is not None:
        dw0_momentum = torch.zeros_like(w0)
        dw1_momentum = torch.zeros_like(w1)
        dw2_momentum = torch.zeros_like(w2)

    output = []

    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  # bf16 [b, l, d]

            lr0i = lr0[:, start:end, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :]  # [b, l, d/1] fp32

            gate_before_act = ki @ w0_now  # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden_before_mul = ki @ w2_now  # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            for _ in range(ttt_update_steps):
                # Fixed objective: neg_dot_product (gradient ascent)
                dhidden = vi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
                dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
                dgate = dhidden * hidden_before_mul
                dgate_before_act = silu_backprop(dgate, gate_before_act)

                w1_grad = zeropower_via_newtonschulz5((hidden * lr1i).transpose(-1, -2) @ vi, muon_update_steps)
                w0_grad = zeropower_via_newtonschulz5(
                    (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
                )
                w2_grad = zeropower_via_newtonschulz5(
                    (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
                )

                if momentum is not None:
                    m_i = momentum[:, start:end, :].mean(dim=1, keepdim=True)
                    w0_grad = w0_grad + dw0_momentum * m_i
                    w1_grad = w1_grad + dw1_momentum * m_i
                    w2_grad = w2_grad + dw2_momentum * m_i
                    dw0_momentum = w0_grad
                    dw1_momentum = w1_grad
                    dw2_momentum = w2_grad

                # Gradient ascent: add gradients
                w1_now = w1_now + w1_grad
                w0_now = w0_now + w0_grad
                w2_now = w2_now + w2_grad

                # do weight norm here
                w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
                w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
                w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            # Only calculate the output in the last repeat.
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)) @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)

    return output, w0, w1, w2


class FastWeightGluMLPMultihead(nn.Module):
    """
    On init of fast_weight:

    Let's start with the magnitude of the value.
    value_proj is initialized with uniform distribution with range [-1.0/sqrt(d), 1.0/sqrt(d)]
        x is layernormed. So during init, value is unit norm total (not per head, per head is 1.0/sqrt(num_head))
        After silu, value is around norm of 2.7 per head.  (why? seems wired)

    Then for the fast weight, assume initial lr = 0.
    Then with l2_norm of q,k, input is unit normed.
    if w0 is initialized with kaiming, relu(w0 @ q) is unit normed.
    Then w1 is initialized with kaiming, so w1 @ relu(w0 @ q) is of norm sqrt(2) per head
    Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
        use_momentum: bool = False,
        ttt_update_steps: int = 1,
        ttt_pre_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps
        self.use_momentum = use_momentum
        self.ttt_update_steps = ttt_update_steps
        self.ttt_pre_norm = ttt_pre_norm

        d_in = d_out = head_dim
        d_h = int(head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )  # [d_in * num_heads,  d_h]
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        # Backward-compatibility for old checkpoints that contain
        # "to_qkv_stack.0.*" even though we now use a fixed single-path forward.
        self.to_qkv_stack = nn.Sequential(nn.Linear(dim, 3 * dim, bias=bias))
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.lr_dim = self.num_heads
        if self.use_momentum:
            self.lr_fc = nn.Linear(dim, self.lr_dim * 3 + 1)
        else:
            self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)

        if self.ttt_pre_norm:
            self.pre_norm = torch.nn.RMSNorm(dim, eps=1e-5, elementwise_affine=True)

        self.o_norm = torch.nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor, info: dict | None = None, *args):
        """
        x: (b, t, l, d) -> (b, t*l, d)
        """
        num_dims = len(x.shape)
        if num_dims == 3:
            x = x.unsqueeze(1)

        b, t, l, d = x.shape  # noqa: E741 - Ambiguous upstream name kept as written.

        if self.ttt_pre_norm:
            x = self.pre_norm(x)

        x = rearrange(x, "b t l d -> b (t l) d")
        qkv = F.silu(self.to_qkv(x), inplace=True)

        q, k, v = rearrange(qkv, "b l (qkv h d) -> qkv (b h) l d", qkv=3, h=self.num_heads)
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5)

        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_fc(x.float())  # [b, l, lr_dim]

        if self.use_momentum:
            momentum = torch.sigmoid(lr[..., -1:])
            lr = lr[..., :-1]
        else:
            momentum = None

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

        lr0, lr1, lr2 = rearrange(lr, "b l (lrs h d) -> lrs (b h) l d", lrs=3, h=self.num_heads)

        if info and "w0" in info and info.get("w0") is not None:
            assert "w1" in info and "w2" in info
            w0 = info["w0"]
            w1 = info["w1"]
            w2 = info["w2"]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0,
            w1,
            w2,
            q,
            k,
            v,
            lr0,
            lr1,
            lr2,
            info["ttt_op_order"] if info else [],
            muon_update_steps=self.muon_update_steps,
            momentum=momentum,
            ttt_update_steps=self.ttt_update_steps,
        )

        output = self.o_norm(output)

        output = rearrange(output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0])

        output = self.c_proj(output)
        output = rearrange(output, "b (t l) d -> b t l d", t=t).to(x.dtype)

        if num_dims == 3:
            output = rearrange(output, "b t l d -> b (t l) d", t=t)

        return output, {
            "w0": w0,
            "w1": w1,
            "w2": w2,
        }

    def extra_repr(self) -> str:
        return (
            f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
            f"Muon update steps: {self.muon_update_steps}, "
            f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, "
        )
