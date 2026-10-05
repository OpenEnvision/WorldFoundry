/**
 * This file includes code originally from the Pytorch Scatter repository:
 * https://github.com/rusty1s/pytorch_scatter
 * Licensed under the MIT License.
 */

#pragma once

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

__device__ __inline__ at::Half __shfl_up_sync(const unsigned mask, const at::Half var, const unsigned int delta) {
    return __shfl_up_sync(mask, var.operator __half(), delta);
}

__device__ __inline__ at::Half __shfl_down_sync(const unsigned mask, const at::Half var, const unsigned int delta) {
    return __shfl_down_sync(mask, var.operator __half(), delta);
}

__device__ __inline__ at::Half __shfl_up(const at::Half var, const unsigned int delta) {
    return __shfl_up(var.operator __half(), delta);
}

__device__ __inline__ at::Half __shfl_down(const at::Half var, const unsigned int delta) {
    return __shfl_down(var.operator __half(), delta);
}

#define SHFL_UP_SYNC __shfl_up_sync
#define SHFL_DOWN_SYNC __shfl_down_sync
