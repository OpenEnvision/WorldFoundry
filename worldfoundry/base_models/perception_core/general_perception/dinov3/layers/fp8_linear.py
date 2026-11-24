# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# FP8 linear layers — optional, requires torch >= 2.4 with FP8 support.
# Import only when FP8 acceleration is explicitly enabled.

try:
    import re

    import torch
    from torch import nn  # noqa: F401 - Import checks optional dependency availability.

    from ..layers.attention import LinearKMaskedBias
    from ..utils import named_replace

    EPS = 1e-12

    def scale(t, amax_t):
        max_v = torch.finfo(torch.float8_e4m3fn).max
        scale_t = torch.clamp(amax_t.float(), min=EPS) / max_v
        t_fp8 = (t / scale_t).to(torch.float8_e4m3fn)
        return t_fp8, scale_t

    def matmul(first, amax_first, second_t, amax_second_t, bias):
        first_fp8, scale_first = scale(first, amax_first)
        second_t_fp8, scale_second_t = scale(second_t, amax_second_t)
        output = torch._scaled_mm(
            first_fp8,
            second_t_fp8.t(),
            scale_a=scale_first.new_ones((1, 1)),
            scale_b=scale_second_t.t().new_ones((1, 1)),
            bias=None,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        output = (output * scale_first * scale_second_t.t()).to(torch.bfloat16)
        if bias is not None:
            output = output + bias
        return output

    @torch.compiler.allow_in_graph
    class Fp8LinearFn(torch.autograd.Function):
        """Fp linear fn implementation."""

        @staticmethod
        def forward(ctx, a, b_t, bias):
            amax_a = a.abs().amax(dim=-1, keepdim=True)
            amax_b_t = b_t.abs().amax(dim=-1, keepdim=True)
            out = matmul(a, amax_a, b_t, amax_b_t, bias)
            ctx.a_requires_grad = a.requires_grad
            ctx.b_requires_grad = b_t.requires_grad
            ctx.bias_requires_grad = bias.requires_grad if bias is not None else False
            ctx.save_for_backward(a, b_t, amax_b_t.max())
            return out

        @staticmethod
        def backward(ctx, grad_out):
            a, b_t, amax_b = ctx.saved_tensors
            if ctx.a_requires_grad:
                b = b_t.t().contiguous()
                amax_grad_out = grad_out.abs().amax(dim=-1, keepdim=True)
                amax_b = amax_b.repeat(b.shape[0], 1)
                grad_a = matmul(grad_out, amax_grad_out, b, amax_b, None)
            else:
                grad_a = None
            grad_b = grad_out.t() @ a if ctx.b_requires_grad else None
            grad_bias = grad_out.sum(dim=0) if ctx.bias_requires_grad else None
            return grad_a, grad_b, grad_bias

    class Fp8Linear(torch.nn.Linear):
        """Fp linear implementation."""

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            out = Fp8LinearFn.apply(input.flatten(end_dim=-2), self.weight, self.bias)
            out = out.unflatten(0, input.shape[:-1])
            return out

    class Fp8LinearKMaskedBias(LinearKMaskedBias):
        """Fp linear k masked bias implementation."""

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            masked_bias = self.bias * self.bias_mask if self.bias is not None else None
            out = Fp8LinearFn.apply(input.flatten(end_dim=-2), self.weight, masked_bias)
            out = out.unflatten(0, input.shape[:-1])
            return out

    def convert_linears_to_fp8(root_module: torch.nn.Module, *, filter: str) -> torch.nn.Module:
        filter_re = re.compile(filter)
        total_count = 0

        def replace(module: torch.nn.Module, name: str) -> torch.nn.Module:
            nonlocal total_count
            if not isinstance(module, torch.nn.Linear) or not filter_re.search(name):
                return module
            if type(module) is torch.nn.Linear:
                new_cls = Fp8Linear
            elif type(module) is LinearKMaskedBias:
                new_cls = Fp8LinearKMaskedBias
            else:
                assert False, str(type(module))  # noqa: B011 - Keeps the assertion semantics under -O.
            new_module = new_cls(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            new_module.weight = module.weight
            new_module.bias = module.bias
            total_count += 1
            return new_module

        out = named_replace(replace, root_module)
        assert total_count > 0, "fp8: no layer found to convert"
        torch._dynamo.reset_code_caches()
        return out

except (ImportError, AttributeError):
    # FP8 not available — provide stub that raises at runtime
    class _FP8Stub:
        """Stub implementation."""

        def __getattr__(self, name):
            raise RuntimeError(
                "FP8 linear layers require torch >= 2.4 with float8_e4m3fn support. "
                "Install a compatible torch version or disable FP8."
            )

    convert_linears_to_fp8 = _FP8Stub()
    Fp8Linear = _FP8Stub()
    Fp8LinearKMaskedBias = _FP8Stub()
