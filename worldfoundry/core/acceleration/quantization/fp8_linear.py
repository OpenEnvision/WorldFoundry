"""Row-scaled Triton FP8 projection provider, including E5M2 by E5M2.

PyTorch scaled_mm rejects the E5M2/E5M2 combination. This explicit provider
retains the common quantization and accounting contract with a native FP8 dot.
"""

import torch

from .linear import Float8Linear


class TritonFloat8Linear(Float8Linear):
    def _forward_quantized(self, input_fp8, input_scale, original_shape, output_dtype):
        if self.scaling != "rowwise" or self.use_fast_accum:
            raise ValueError("Triton FP8 projections require rowwise scales and FP32 accumulation")
        if torch.cuda.get_device_capability(input_fp8.device)[0] < 9:
            raise ValueError("Triton FP8 projections require NVIDIA SM90 or newer")
        from .triton_fp8_linear import fp8_linear

        output = fp8_linear(input_fp8, self.weight_fp8, input_scale, self.weight_scale, self.bias, output_dtype)
        self.low_precision_kernel_calls += 1
        self.request_low_precision_kernel_calls += 1
        return output.reshape(*original_shape[:-1], self.out_features)

    def runtime_report(self):
        return {**super().runtime_report(), "provider": "triton-rowwise-fp8"}
