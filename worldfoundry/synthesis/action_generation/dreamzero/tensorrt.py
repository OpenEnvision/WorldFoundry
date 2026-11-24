# Inference-only DreamZero runtime retained in-tree.
import atexit
import ctypes

import tensorrt as trt
import torch


def torch_type(trt_type):
    mapping = {
        trt.float32: torch.float32,  # Added missing FLOAT mapping
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
    }
    if trt_type in mapping:
        return mapping[trt_type]

    raise TypeError(f"Could not resolve TensorRT datatype to an equivalent torch datatype. {trt_type}")


class Engine:
    def __init__(self, file, plugins=()):
        super().__init__()

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        self.plugins = [ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL) for plugin in plugins]
        self.file = file
        self.load(file)

        def destroy(self):
            del self.execution_context
            del self.handle

        atexit.register(destroy, self)
        self.print()

    def print(self):
        print("============= TRT Engine Detail =============")
        print(f"Engine file: {self.file}")
        print(f"Inputs: {len(self.in_meta)}")
        for ib, item in enumerate(self.in_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")

        print(f"Outputs: {len(self.out_meta)}")
        for ib, item in enumerate(self.out_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")
        print("=============================================")

    def load(self, file):
        runtime = trt.Runtime(self.logger)

        with open(file, "rb") as f:
            self.handle = runtime.deserialize_cuda_engine(f.read())
            assert self.handle is not None, f"Failed to deserialize the cuda engine from file: {file}"

        self.execution_context = self.handle.create_execution_context()
        self.meta, self.in_meta, self.out_meta = [], [], []
        for tensor_name in self.handle:
            shape = self.handle.get_tensor_shape(tensor_name)
            print(f"Tensor name: {tensor_name}, shape: {shape}")
            dtype = torch_type(self.handle.get_tensor_dtype(tensor_name))
            if self.handle.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.in_meta.append([tensor_name, shape, dtype])
            else:
                self.out_meta.append([tensor_name, shape, dtype])

    def __call__(self, *args, **inputs):
        return self.forward(*args, **inputs)

    def set_runtime_tensor_shape(self, name, shape):
        self.execution_context.set_input_shape(name, shape)

    def forward(self, *args, **kwargs):
        return_list = kwargs.pop("return_list", False)
        reference_tensors = []
        stream = torch.cuda.current_stream()
        for iarg, x in enumerate(args):
            name, shape, dtype = self.in_meta[iarg]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            assert isinstance(x, torch.Tensor), f"Unsupported tensor type: {type(x)}"
            assert runtime_shape == x.shape, f"Invalid input shape: {runtime_shape} != {x.shape}"
            assert dtype == x.dtype, f"Invalid tensor dtype, excepted dtype is {dtype}, but got {x.dtype}"
            assert x.is_cuda, f"Invalid tensor device, excepted device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for name, _shape, dtype in self.in_meta:
            if name not in kwargs:
                continue

            runtime_shape = self.execution_context.get_tensor_shape(name)
            x = kwargs[name]
            assert isinstance(x, torch.Tensor), f"Unsupported tensor[{name}] type: {type(x)}"
            assert runtime_shape == x.shape, (
                f"Invalid input[{name}] shape: {x.shape}, but the expected shape is: {runtime_shape}"
            )
            assert dtype == x.dtype, f"Invalid tensor[{name}] dtype, expected dtype is {dtype}, but got {x.dtype}"
            assert x.is_cuda, f"Invalid tensor[{name}] device, expected device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for item in self.out_meta:
            name = item[0]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            output_tensor = torch.zeros(*runtime_shape, dtype=item[2], device=reference_tensors[0].device)
            self.execution_context.set_tensor_address(name, output_tensor.data_ptr())
            reference_tensors.append(output_tensor)

        self.execution_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        assert len(reference_tensors) == len(self.in_meta) + len(self.out_meta), (
            f"Invalid input tensors. The expected I/O tensors are {len(self.in_meta) + len(self.out_meta)}, but got {len(reference_tensors)}"
        )

        if return_list:
            return [reference_tensors[len(self.in_meta) + i] for i, item in enumerate(self.out_meta)]
        else:
            return {item[0]: reference_tensors[len(self.in_meta) + i] for i, item in enumerate(self.out_meta)}


class WanTrtModel5B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
    ):
        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)

        output = self.engine(
            x=x.to(torch.float16),
            action=action.to(torch.float16),
            timestep=timestep.to(torch.float16),
            context=context.to(torch.float16),
            state=state.to(torch.float16),
            embodiment_id=embodiment_id.to(torch.int32),
        )
        if "out.0" in output:  # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(
                torch.bfloat16
            ).contiguous()


class WanTrtModel14B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        clip_feature: torch.Tensor,
        y: torch.Tensor,
    ):
        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)
        self.engine.set_runtime_tensor_shape("clip_feature", clip_feature.shape)
        self.engine.set_runtime_tensor_shape("y", y.shape)

        output = self.engine(
            x=x.to(torch.float16),
            action=action.to(torch.float16),
            timestep=timestep.to(torch.float16),
            context=context.to(torch.float16),
            state=state.to(torch.float16),
            embodiment_id=embodiment_id.to(torch.int32),
            clip_feature=clip_feature.to(torch.float16),
            y=y.to(torch.float16),
        )
        if "out.0" in output:  # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(
                torch.bfloat16
            ).contiguous()


class WanTrtModelAr5B(torch.nn.Module):
    """TRT wrapper for ar_5B_n6 model type - uses kv_cache but no clip_feature."""

    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x,
        timestep,
        context,
        kv_cache: list[torch.Tensor],
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
    ):
        del y, clip_feature

        kv_cache_packed = torch.stack(kv_cache, dim=0)

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("timestep", timestep.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("kv_cache_packed", kv_cache_packed.shape)
        # self.engine.set_runtime_tensor_shape("y", y.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("timestep_action", timestep_action.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)

        output = self.engine(
            x.to(torch.float16),
            timestep.to(torch.float16),
            context.to(torch.float16),
            kv_cache_packed.to(torch.float16),
            # y.to(torch.float16),
            action.to(torch.float16),
            timestep_action.to(torch.float16),
            state.to(torch.float16),
        )

        if "out.0" in output:  # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(
                torch.bfloat16
            ).contiguous()


class WanTrtModelAr14B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x,
        timestep,
        context,
        kv_cache: list[torch.Tensor],
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
    ):
        kv_cache_packed = torch.stack(kv_cache, dim=0)

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("timestep", timestep.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("kv_cache_packed", kv_cache_packed.shape)
        self.engine.set_runtime_tensor_shape("y", y.shape)
        self.engine.set_runtime_tensor_shape("clip_feature", clip_feature.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("timestep_action", timestep_action.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)

        output = self.engine(
            x.to(torch.float16),
            timestep.to(torch.float16),
            context.to(torch.float16),
            kv_cache_packed.to(torch.float16),
            y.to(torch.float16),
            clip_feature.to(torch.float16),
            action.to(torch.float16),
            timestep_action.to(torch.float16),
            state.to(torch.float16),
        )

        if "out.0" in output:  # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(
                torch.bfloat16
            ).contiguous()


def load_tensorrt_engine(engine_path="tensorrt/wan_model.trt", model_type="5B"):
    if model_type == "5B":
        trt_inference = WanTrtModel5B(engine_path)
    elif model_type == "ar_5B_n6" or model_type == "ar_5B":
        trt_inference = WanTrtModelAr5B(engine_path)
    elif model_type == "14B":
        trt_inference = WanTrtModel14B(engine_path)
    elif model_type == "ar_14B" or model_type == "ar_14B_droid":
        trt_inference = WanTrtModelAr14B(engine_path)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    return trt_inference
