"""Per-module VRAM wrappers: offload / onload / compute-device state machine.

:class:`AutoTorchModule` and its Linear/Module subclasses wrap a child so
inactive weights sit on ``offload_device`` (often CPU or a :class:`DiskMap`
view) and move to ``computation_device`` only around ``forward``. State
``0/1/2`` is offloaded / onloaded / pinned-for-compute.

Why wrap instead of ``model.cpu()`` / ``model.cuda()``: a video DiT has
dozens of blocks; moving the whole tree every step thrashes the allocator.
Wrappers move *one* module and can skip the copy when VRAM headroom
(``vram_limit``) says the weight is already resident.

``WORLDFOUNDRY_VRAM_CHECK_INTERVAL`` throttles free-memory probes — checking
every layer on every step is itself a CUDA sync. Deep-copy fallbacks log
once per class because a misplaced preparing/compute device pair otherwise
silently clones a 100M-parameter block each call.
"""

import copy
import logging
import os
from typing import Union

import torch

from worldfoundry.core.execution.device import get_device_name, is_npu_available, parse_device_type

from .disk_map import DiskMap
from .initialization import init_weights_on_device, skip_model_initialization

_FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
    )
    if dtype is not None
)
logger = logging.getLogger(__name__)
_DEEPCOPY_WARNING_CLASSES: set[type[torch.nn.Module]] = set()
_INVALID_VRAM_CHECK_INTERVALS: set[str] = set()

# ──────────────────────────────────────────────────────────────────────────
# Probe throttle and deepcopy warning — mem_get_info is a CUDA sync
# ──────────────────────────────────────────────────────────────────────────


def _vram_check_interval() -> int:
    """Return how many ``check_free_vram`` calls may reuse the last probe.

    ``mem_get_info`` synchronizes the device. Checking every layer every step
    is itself a stall. Invalid env values fail closed to ``1`` (probe every
    check) and log once per distinct bad string.
    """
    value = os.getenv("WORLDFOUNDRY_VRAM_CHECK_INTERVAL", "1")
    try:
        interval = int(value)
    except ValueError:
        interval = 1
    if interval < 1:
        interval = 1
    if str(interval) != value and value not in _INVALID_VRAM_CHECK_INTERVALS:
        _INVALID_VRAM_CHECK_INTERVALS.add(value)
        logger.warning(
            "Invalid WORLDFOUNDRY_VRAM_CHECK_INTERVAL=%r; using 1 (probe every check).",
            value,
        )
    return interval


def _warn_deepcopy_once(module: torch.nn.Module) -> None:
    """Log once per class when a wrapper deep-copies a module for a compute cast.

    A misplaced preparing/compute device pair otherwise silently clones a
    100M-parameter block on every forward.
    """
    module_type = type(module)
    if module_type in _DEEPCOPY_WARNING_CLASSES:
        return
    _DEEPCOPY_WARNING_CLASSES.add(module_type)
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    logger.warning(
        "Temporary VRAM casting deep-copies %s (%d parameters) on each computation; "
        "align preparing/computation placement or use a specialized wrapper to avoid this cost.",
        module_type.__name__,
        parameter_count,
    )


# ──────────────────────────────────────────────────────────────────────────
# State machine — 0 offloaded / 1 onloaded / 2 pinned for compute
# ──────────────────────────────────────────────────────────────────────────


class AutoTorchModule(torch.nn.Module):
    """Base state machine for modules that move between memory tiers.

    ``state`` is ``0`` while offloaded, ``1`` while onloaded, and ``2`` while
    kept on the computation device. Subclasses decide whether transitions copy
    tensors, materialize them from disk, or use temporary computation weights.
    """

    def __init__(
        self,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
    ):
        """Configure the dtype and device used at each lifecycle stage.

        Args:
            offload_dtype: Storage dtype while the module is inactive.
            offload_device: Storage device while inactive; subclasses may also
                accept the sentinel ``"disk"``.
            onload_dtype: Dtype after the first prefetch transition.
            onload_device: Device used for prefetched weights.
            preparing_dtype: Dtype used by the optional second prefetch stage.
            preparing_device: Device used by the preparing stage.
            computation_dtype: Dtype used for the actual forward operation.
            computation_device: Device used for the actual forward operation.
            vram_limit: Optional used-memory limit in GiB. Below the limit,
                wrappers may keep prepared weights resident.
        """
        super().__init__()
        self.set_dtype_and_device(
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
        )
        self.state = 0
        self.name = ""
        self.computation_device_type = parse_device_type(self.computation_device)

    def set_dtype_and_device(
        self,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
    ):
        """Update lifecycle placement, defaulting omitted stages to computation placement."""
        self.offload_dtype = offload_dtype or computation_dtype
        self.offload_device = offload_device or computation_device
        self.onload_dtype = onload_dtype or computation_dtype
        self.onload_device = onload_device or computation_device
        self.preparing_dtype = preparing_dtype or computation_dtype
        self.preparing_device = preparing_device or computation_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit
        self._vram_check_interval = _vram_check_interval()
        self._vram_checks_since_probe = 0
        self._cached_vram_decision: bool | None = None

    def cast_to(self, weight, dtype, device):
        """Copy one tensor to ``dtype`` and ``device`` without mutating the source."""
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight)
        return r

    def check_free_vram(self):
        """Return whether current accelerator usage is below ``vram_limit``."""
        if self.vram_limit is None or self.computation_device_type not in {"cuda", "npu"}:
            return True
        if (
            self._cached_vram_decision is not None
            and self._vram_checks_since_probe < self._vram_check_interval - 1
        ):
            self._vram_checks_since_probe += 1
            return self._cached_vram_decision
        device = self.computation_device if not is_npu_available() else get_device_name()
        gpu_mem_state = getattr(torch, self.computation_device_type).mem_get_info(device)
        used_memory = (gpu_mem_state[1] - gpu_mem_state[0]) / (1024**3)
        self._cached_vram_decision = used_memory < self.vram_limit
        self._vram_checks_since_probe = 0
        return self._cached_vram_decision

    def offload(self):
        """Move managed parameters to the inactive placement and set state 0."""
        if self.state != 0:
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        """Move managed parameters to the first prefetch placement and set state 1."""
        if self.state != 1:
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def keep(self):
        """Keep managed parameters at computation placement and set state 2."""
        if self.state != 2:
            self.to(dtype=self.computation_dtype, device=self.computation_device)
            self.state = 2

    def param_name(self, name):
        """Return the fully qualified checkpoint key for a local parameter name."""
        if self.name == "":
            return name
        else:
            return self.name + "." + name


# ──────────────────────────────────────────────────────────────────────────
# Recursive wrap — disk vs device offload; deepcopy only when placement differs
# ──────────────────────────────────────────────────────────────────────────


class AutoWrappedModule(AutoTorchModule):
    """Wrap an arbitrary module with staged CPU/GPU/disk weight movement.

    Forward calls opportunistically prepare weights, materialize a computation
    copy only when placement differs, and then delegate to the wrapped module.
    Attribute access falls through to the wrapped module so model code can keep
    using its original interface.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
        name: str = "",
        disk_map: DiskMap = None,
        **kwargs,
    ):
        """Wrap ``module`` and bind it to a placement policy.

        Args:
            module: Original PyTorch module.
            offload_dtype: Inactive storage dtype or ``"disk"``.
            offload_device: Inactive storage device or ``"disk"``.
            onload_dtype: First-stage prefetch dtype.
            onload_device: First-stage prefetch device.
            preparing_dtype: Second-stage prefetch dtype.
            preparing_device: Second-stage prefetch device.
            computation_dtype: Forward-pass dtype.
            computation_device: Forward-pass device.
            vram_limit: Optional used-memory limit in GiB.
            name: Qualified module name used to resolve disk-map keys.
            disk_map: Lazy checkpoint tensor mapping required for disk offload.
            **kwargs: Reserved for wrapper-compatible module maps.
        """
        super().__init__(
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
        )
        self.module = module
        if offload_dtype == "disk":
            self.name = name
            self.disk_map = disk_map
            self.required_params = [name for name, _ in self.module.named_parameters()]
            self.disk_offload = True
        else:
            self.disk_offload = False

    def load_from_disk(self, torch_dtype, device, copy_module=False):
        """Materialize disk-backed parameters onto *device*; buffers stay resident.

        Non-persistent buffers cannot be reconstructed from a state dict, so
        they are never replaced. ``copy_module=True`` clones first so the
        stored module can remain on meta after a temporary compute load.
        """
        if copy_module:
            module = copy.deepcopy(self.module)
        else:
            module = self.module
        state_dict = {}
        for name in self.required_params:
            param = self.disk_map[self.param_name(name)]
            param = param.to(dtype=torch_dtype, device=device)
            state_dict[name] = param
        # Parameters are disk-backed, but buffers deliberately remain resident
        # because non-persistent buffers cannot be reconstructed from a state
        # dict. Missing buffer keys are therefore expected here.
        module.load_state_dict(state_dict, assign=True, strict=False)
        module.to(dtype=torch_dtype, device=device)
        return module

    def offload_to_disk(self, model: torch.nn.Module):
        """Replace disk-backed parameters with meta placeholders, retaining buffers."""

        direct_buffers = tuple(model.named_buffers(recurse=False))
        if direct_buffers:
            logger.debug(
                "Retaining %d direct buffer(s) on %s while offloading its parameters to disk.",
                len(direct_buffers),
                type(model).__name__,
            )
        for name, parameter in tuple(model.named_parameters(recurse=False)):
            model._parameters[name] = torch.nn.Parameter(
                torch.empty_like(parameter, device="meta"),
                requires_grad=parameter.requires_grad,
            )
        for child in model.children():
            self.offload_to_disk(child)

    def offload(self):
        """Return to inactive placement: meta placeholders on disk, else ``.to(offload)``.

        Disk offload must not ``.to(cpu)`` — that would materialize the full
        module and defeat the mmap path.
        """
        # offload / onload / preparing -> offload
        if self.state != 0:
            if self.disk_offload:
                self.offload_to_disk(self.module)
            else:
                self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        """Prefetch from disk or move from offload; skip when the target is still disk."""
        # offload / onload / preparing -> onload
        if self.state < 1:
            if self.disk_offload and self.onload_device != "disk" and self.offload_device == "disk":
                self.load_from_disk(self.onload_dtype, self.onload_device)
            elif self.onload_device != "disk":
                self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def preparing(self):
        """Second-stage prefetch; materialize only when leaving disk for a real device."""
        # onload / preparing -> preparing
        if self.state != 2:
            if self.disk_offload and self.preparing_device != "disk" and self.onload_device == "disk":
                self.load_from_disk(self.preparing_dtype, self.preparing_device)
            elif self.preparing_device != "disk":
                self.to(dtype=self.preparing_dtype, device=self.preparing_device)
            self.state = 2

    def cast_to(self, module, dtype, device):
        """Deep-copy then move so the stored offload weights are not mutated in place."""
        _warn_deepcopy_once(module)
        return copy.deepcopy(module).to(dtype=dtype, device=device)

    def computation(self):
        """Return a module at compute placement without permanently leaving the current state.

        Reuses ``self.module`` when dtype/device already match. Disk + copy
        keeps the stored module on meta. Otherwise :meth:`cast_to` clones.
        """
        # offload / onload / preparing -> computation (temporary)
        if self.state == 2:
            torch_dtype, device = self.preparing_dtype, self.preparing_device
        elif self.state == 1:
            torch_dtype, device = self.onload_dtype, self.onload_device
        else:
            torch_dtype, device = self.offload_dtype, self.offload_device
        if torch_dtype == self.computation_dtype and device == self.computation_device:
            module = self.module
        elif self.disk_offload and device == "disk":
            module = self.load_from_disk(self.computation_dtype, self.computation_device, copy_module=True)
        else:
            module = self.cast_to(self.module, dtype=self.computation_dtype, device=self.computation_device)
        return module

    def forward(self, *args, **kwargs):
        """Promote to preparing when VRAM headroom allows, then run a compute copy."""
        if self.state == 1 and (self.vram_limit is None or self.check_free_vram()):
            self.preparing()
        module = self.computation()
        return module(*args, **kwargs)

    def __getattr__(self, name):
        """Fall through to the wrapped module so callers keep the original interface."""
        if name in self.__dict__ or name == "module":
            return super().__getattr__(name)
        else:
            return getattr(self.module, name)


# ──────────────────────────────────────────────────────────────────────────
# Direct-parameter wrap — children are wrapped separately; never pickle DiskMap
# ──────────────────────────────────────────────────────────────────────────


class AutoWrappedNonRecurseModule(AutoWrappedModule):
    """Manage only a module's direct parameters while its children are wrapped separately."""

    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
        name: str = "",
        disk_map: DiskMap = None,
        **kwargs,
    ):
        """Wrap *module* and restrict disk keys to direct parameters only.

        Children are wrapped separately; listing recursive names would
        double-load those weights and fight the child wrappers.
        """
        super().__init__(
            module,
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
            name,
            disk_map,
            **kwargs,
        )
        if self.disk_offload:
            self.required_params = [name for name, _ in self.module.named_parameters(recurse=False)]

    def load_from_disk(self, torch_dtype, device, copy_module=False):
        """Materialize this module's direct parameters only; children own their own maps."""
        if copy_module:
            module = copy.deepcopy(self.module)
        else:
            module = self.module
        state_dict = {}
        for name in self.required_params:
            param = self.disk_map[self.param_name(name)]
            param = param.to(dtype=torch_dtype, device=device)
            state_dict[name] = param
        module.load_state_dict(state_dict, assign=True, strict=False)
        return module

    def offload_to_disk(self, model: torch.nn.Module):
        """Replace only direct parameters with meta placeholders; children stay untouched."""
        for name in self.required_params:
            parameter = model._parameters.get(name)
            if parameter is None:
                continue
            model._parameters[name] = torch.nn.Parameter(
                torch.empty_like(parameter, device="meta"),
                requires_grad=parameter.requires_grad,
            )

    def cast_to(self, module, dtype, device):
        """Return *module* unchanged; the architecture owns parameter casts.

        Deep-copying would pickle child ``DiskMap`` readers (open safetensors
        handles) and fail. Callers must already be at compute placement.
        """
        # Parameter casting is implemented in the model architecture.
        return module

    def forward(self, *args, **kwargs):
        """In-place disk materialize when offloaded; otherwise the recursive path.

        Deep-copy of a hierarchy that contains independently disk-backed
        children tries to pickle their open safetensors readers.
        """
        if self.disk_offload and self.state == 0:
            # A non-recursive wrapper may contain independently disk-backed
            # children.  Deep-copying that hierarchy attempts to pickle their
            # open safetensors readers, so materialize only this module's
            # direct parameters in place for the duration of the call.
            module = self.load_from_disk(
                self.computation_dtype,
                self.computation_device,
                copy_module=False,
            )
            try:
                return module(*args, **kwargs)
            finally:
                self.offload_to_disk(module)
        return super().forward(*args, **kwargs)

    def __getattr__(self, name):
        """Fall through to the wrapped module; ``module`` itself stays on this class."""
        if name in self.__dict__ or name == "module":
            return super().__getattr__(name)
        else:
            return getattr(self.module, name)


# ──────────────────────────────────────────────────────────────────────────
# LayerNorm / Linear specialists — avoid deepcopy; Linear owns FP8 and LoRA
# ──────────────────────────────────────────────────────────────────────────


class WanAutoCastLayerNorm(torch.nn.LayerNorm, AutoTorchModule):
    """LayerNorm adapter that materializes affine weights at computation precision."""

    def __init__(
        self,
        module: torch.nn.LayerNorm,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
        **kwargs,
    ):
        """Steal affine tensors from *module* after a meta construction.

        Building a second LayerNorm on the default device would double-allocate
        the affine weights. Meta init plus assignment keeps one storage.
        """
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                bias=module.bias is not None,
                dtype=offload_dtype,
                device=offload_device,
            )
        self.set_dtype_and_device(
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
        )
        self.weight = module.weight
        self.bias = module.bias
        self.state = 0
        self.computation_device_type = parse_device_type(self.computation_device)

    def forward(self, x, *args, **kwargs):
        """Norm in float32 then ``type_as(x)`` so mixed-precision affine stays stable.

        Temporary casts are used when VRAM headroom is tight; otherwise
        :meth:`keep` pins weights at compute placement for the next step.
        """
        if self.state == 2:
            weight, bias = self.weight, self.bias
        else:
            if self.state == 1:
                source_dtype, source_device = self.onload_dtype, self.onload_device
            else:
                source_dtype, source_device = self.offload_dtype, self.offload_device
            if source_dtype == self.computation_dtype and source_device == self.computation_device:
                weight, bias = self.weight, self.bias
            elif self.vram_limit is not None and self.check_free_vram():
                self.keep()
                weight, bias = self.weight, self.bias
            else:
                weight = (
                    None
                    if self.weight is None
                    else self.cast_to(self.weight, self.computation_dtype, self.computation_device)
                )
                bias = (
                    None
                    if self.bias is None
                    else self.cast_to(self.bias, self.computation_dtype, self.computation_device)
                )
        with torch.amp.autocast(device_type=x.device.type):
            return torch.nn.functional.layer_norm(
                x.float(),
                self.normalized_shape,
                weight,
                bias,
                self.eps,
            ).type_as(x)


class AutoWrappedLinear(torch.nn.Linear, AutoTorchModule):
    """Linear-layer wrapper with staged placement, disk loading, FP8, and LoRA support."""

    def __init__(
        self,
        module: torch.nn.Linear,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
        name: str = "",
        disk_map: DiskMap = None,
        **kwargs,
    ):
        """Wrap an existing linear layer with one placement policy.

        Args:
            module: Source ``torch.nn.Linear`` whose parameters are reused.
            offload_dtype: Inactive storage dtype or ``"disk"``.
            offload_device: Inactive storage device or ``"disk"``.
            onload_dtype: First prefetch dtype.
            onload_device: First prefetch device.
            preparing_dtype: Second prefetch dtype.
            preparing_device: Second prefetch device.
            computation_dtype: Matmul dtype; supported FP8 dtypes use scaled MM.
            computation_device: Matmul device.
            vram_limit: Optional used-memory limit in GiB.
            name: Qualified checkpoint prefix for weight and bias.
            disk_map: Required lazy tensor map when disk offload is selected.
            **kwargs: Reserved for module-map compatibility.
        """
        with skip_model_initialization():
            super().__init__(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        self.set_dtype_and_device(
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
        )
        self.weight = module.weight
        self.bias = module.bias
        self.state = 0
        self.name = name
        self.lora_A_weights = []
        self.lora_B_weights = []
        self.lora_merger = None
        self.enable_fp8 = computation_dtype in _FLOAT8_DTYPES
        self.computation_device_type = parse_device_type(self.computation_device)

        if offload_dtype == "disk":
            self.disk_map = disk_map
            self.disk_offload = True
        else:
            self.disk_offload = False

    def fp8_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """Row-wise scaled FP8 matmul via ``torch._scaled_mm``.

        Scales are the exact values passed to the kernel so dequantization
        matches the quantization (CC-28). ``e4m3fnuz`` halves the representable
        max; that factor is baked into ``fp8_max`` and recovered by the kernel
        scales, not a post-hoc multiply.
        """
        device = input.device
        origin_dtype = input.dtype
        origin_shape = input.shape
        input = input.reshape(-1, origin_shape[-1])

        x_max = torch.max(torch.abs(input), dim=-1, keepdim=True).values
        fp8_max = 448.0
        # For float8_e4m3fnuz, the maximum representable value is half of that of e4m3fn.
        # To avoid overflow and ensure numerical compatibility during FP8 computation,
        # we scale down the input by 2.0 in advance.
        # This scaling will be compensated later during the final result scaling.
        if self.computation_dtype == getattr(torch, "float8_e4m3fnuz", None):
            fp8_max = fp8_max / 2.0
        scale_a = torch.clamp(x_max / fp8_max, min=1.0).float().to(device=device)
        weight_max = torch.amax(torch.abs(weight), dim=1, keepdim=True)
        scale_b = torch.clamp(weight_max / fp8_max, min=1.0).float().to(device=device)
        # Both operands use the exact scales handed to _scaled_mm. The former
        # unit weight scale silently saturated values outside the FP8 range;
        # adding epsilon only on the input division also made dequantization
        # mathematically inconsistent (CC-28).
        input = input / scale_a
        weight = weight / scale_b
        input = input.to(self.computation_dtype)
        weight = weight.to(self.computation_dtype)
        bias = None if bias is None else bias.to(torch.bfloat16)

        result = torch._scaled_mm(
            input,
            weight.T,
            scale_a=scale_a,
            scale_b=scale_b.T,
            bias=bias,
            out_dtype=origin_dtype,
        )
        new_shape = origin_shape[:-1] + result.shape[-1:]
        result = result.reshape(new_shape)
        return result

    def load_from_disk(self, torch_dtype, device, assign=True):
        """Fetch ``name.weight`` / ``name.bias`` from the disk map.

        ``assign=False`` returns tensors without mutating this module so a
        temporary compute copy can leave the stored weights on meta.
        """
        weight = self.disk_map[self.name + ".weight"].to(dtype=torch_dtype, device=device)
        bias = None if self.bias is None else self.disk_map[self.name + ".bias"].to(dtype=torch_dtype, device=device)
        if assign:
            state_dict = {"weight": weight}
            if bias is not None:
                state_dict["bias"] = bias
            self.load_state_dict(state_dict, assign=True)
        return weight, bias

    def offload(self):
        """Drop to meta on disk offload; otherwise move this Linear to offload placement."""
        # offload / onload / preparing -> offload
        if self.state != 0:
            if self.disk_offload:
                self.to("meta")
            else:
                self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        """Prefetch this Linear from disk or move from offload; skip a still-disk target."""
        # offload / onload / preparing -> onload
        if self.state < 1:
            if self.disk_offload and self.onload_device != "disk" and self.offload_device == "disk":
                self.load_from_disk(self.onload_dtype, self.onload_device)
            elif self.onload_device != "disk":
                self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def preparing(self):
        """Second-stage prefetch for this Linear; materialize only when leaving disk."""
        # onload / preparing -> preparing
        if self.state != 2:
            if self.disk_offload and self.preparing_device != "disk" and self.onload_device == "disk":
                self.load_from_disk(self.preparing_dtype, self.preparing_device)
            elif self.preparing_device != "disk":
                self.to(dtype=self.preparing_dtype, device=self.preparing_device)
            self.state = 2

    def computation(self):
        """Return weight/bias at compute placement without permanently leaving the current state.

        ``assign=False`` on the disk path so the stored Linear stays on meta.
        """
        # offload / onload / preparing -> computation (temporary)
        if self.state == 2:
            torch_dtype, device = self.preparing_dtype, self.preparing_device
        elif self.state == 1:
            torch_dtype, device = self.onload_dtype, self.onload_device
        else:
            torch_dtype, device = self.offload_dtype, self.offload_device
        if torch_dtype == self.computation_dtype and device == self.computation_device:
            weight, bias = self.weight, self.bias
        elif self.disk_offload and device == "disk":
            weight, bias = self.load_from_disk(self.computation_dtype, self.computation_device, assign=False)
        else:
            weight = self.cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = (
                None if self.bias is None else self.cast_to(self.bias, self.computation_dtype, self.computation_device)
            )
        return weight, bias

    def linear_forward(self, x, weight, bias):
        """Dense GEMM, or the FP8 scaled path when ``computation_dtype`` is float8."""
        if self.enable_fp8:
            out = self.fp8_linear(x, weight, bias)
        else:
            out = torch.nn.functional.linear(x, weight, bias)
        return out

    def lora_forward(self, x, out):
        """Add LoRA deltas after the base linear so merge order matches PEFT (B @ A).

        A custom ``lora_merger`` stacks adapters instead of summing, for
        multi-adapter blends that are not a simple scale.
        """
        if self.lora_merger is None:
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                out = out + x @ lora_A.T @ lora_B.T
        else:
            lora_output = []
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                lora_output.append(x @ lora_A.T @ lora_B.T)
            lora_output = torch.stack(lora_output)
            out = self.lora_merger(out, lora_output)
        return out

    def forward(self, x, *args, **kwargs):
        """Promote when VRAM allows, run the Linear, then apply LoRA on the output.

        LoRA is applied after the base GEMM so the merge order matches PEFT
        (delta on the activation, not a second fused weight).
        """
        if self.state == 1 and (self.vram_limit is None or self.check_free_vram()):
            self.preparing()
        weight, bias = self.computation()
        out = self.linear_forward(x, weight, bias)
        if len(self.lora_A_weights) > 0:
            out = self.lora_forward(x, out)
        return out


# ──────────────────────────────────────────────────────────────────────────
# Install wrappers — wrap vs whole-model .to(); overflow policy after param budget
# ──────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def move_direct_tensors_to_device(
    module: torch.nn.Module,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """Move only a module's own parameters and buffers.

    Fine-grained VRAM wrapping replaces selected descendants while leaving
    parameters registered directly on their parent untouched.  Architectures
    such as Wan's CLIP vision tower keep positional/class embeddings and the
    output projection directly on the root module, so those small tensors must
    follow the computation device without recursively moving the wrapped
    transformer blocks.
    """

    target_device = torch.device(device)
    for name, parameter in tuple(module.named_parameters(recurse=False)):
        target_dtype = dtype if dtype is not None and parameter.is_floating_point() else parameter.dtype
        module._parameters[name] = torch.nn.Parameter(
            parameter.to(device=target_device, dtype=target_dtype),
            requires_grad=parameter.requires_grad,
        )
    for name, buffer in tuple(module.named_buffers(recurse=False)):
        target_dtype = dtype if dtype is not None and buffer.is_floating_point() else buffer.dtype
        module._buffers[name] = buffer.to(device=target_device, dtype=target_dtype)
    return module


def enable_vram_management_recursively(
    model: torch.nn.Module,
    module_map: dict,
    vram_config: dict,
    vram_limit=None,
    name_prefix="",
    disk_map=None,
    max_num_param=None,
    overflow_vram_config: dict | None = None,
    total_num_param=0,
    **kwargs,
):
    """Replace matching descendants with lifecycle-aware wrapper modules.

    Args:
        model: Root module mutated in place.
        module_map: Mapping from source module classes to wrapper classes.
        vram_config: Default offload/onload/preparing/computation placements.
        vram_limit: Optional used-memory limit in GiB forwarded to wrappers.
        name_prefix: Prefix used when resolving checkpoint keys through a
            ``DiskMap``.
        disk_map: Optional lazy checkpoint mapping for disk-backed wrappers.
        max_num_param: Parameter budget that switches later modules to
            ``overflow_vram_config``.
        overflow_vram_config: Placement policy used after the parameter budget.
        total_num_param: Running parameter count for recursive calls.
        **kwargs: Extra wrapper constructor arguments.

    Returns:
        Total number of parameters replaced at or below this module.

    Notes:
        The function mutates child modules. Most callers should use
        ``enable_vram_management`` so the root itself is handled correctly and
        the ``vram_management_enabled`` marker is installed.
    """
    if isinstance(model, AutoWrappedNonRecurseModule):
        model = model.module
    for name, module in model.named_children():
        layer_name = name if name_prefix == "" else name_prefix + "." + name
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                selected_vram_config = vram_config
                if (
                    max_num_param is not None
                    and overflow_vram_config is not None
                    and total_num_param + num_param > max_num_param
                ):
                    selected_vram_config = overflow_vram_config
                module_ = target_module(
                    module, **selected_vram_config, vram_limit=vram_limit, name=layer_name, disk_map=disk_map, **kwargs
                )
                if isinstance(module_, AutoWrappedNonRecurseModule):
                    enable_vram_management_recursively(
                        module_,
                        module_map,
                        selected_vram_config,
                        vram_limit=vram_limit,
                        name_prefix=layer_name,
                        disk_map=disk_map,
                        max_num_param=max_num_param,
                        overflow_vram_config=overflow_vram_config,
                        total_num_param=total_num_param,
                        **kwargs,
                    )
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(
                module,
                module_map,
                vram_config,
                vram_limit=vram_limit,
                name_prefix=layer_name,
                disk_map=disk_map,
                max_num_param=max_num_param,
                overflow_vram_config=overflow_vram_config,
                total_num_param=total_num_param,
                **kwargs,
            )
    return total_num_param


def fill_vram_config(model, vram_config):
    """Collapse a placement policy when the root module is wrapped as one unit."""
    vram_config_ = vram_config.copy()
    vram_config_["onload_dtype"] = vram_config["computation_dtype"]
    vram_config_["onload_device"] = vram_config["computation_device"]
    vram_config_["preparing_dtype"] = vram_config["computation_dtype"]
    vram_config_["preparing_device"] = vram_config["computation_device"]
    for k in vram_config:
        if vram_config[k] != vram_config_[k]:
            print(
                f"No fine-grained VRAM configuration is provided for {model.__class__.__name__}. [`onload`, `preparing`, `computation`] will be the same state. `vram_config` is set to {vram_config_}"
            )
            break
    return vram_config_


def enable_vram_management(
    model: torch.nn.Module,
    module_map: dict,
    vram_config: dict,
    vram_limit=None,
    disk_map=None,
    max_num_param=None,
    overflow_vram_config: dict | None = None,
    **kwargs,
):
    """Install staged VRAM management on a model and return that model.

    Args:
        model: Model to wrap. It may be replaced when its own class appears in
            ``module_map``; otherwise matching descendants are mutated.
        module_map: Mapping from original module classes to wrapper classes,
            for example ``{torch.nn.Linear: AutoWrappedLinear}``.
        vram_config: Dictionary containing offload, onload, preparing, and
            computation dtype/device pairs.
        vram_limit: Optional used-memory limit in GiB.
        disk_map: Lazy checkpoint mapping used by disk-backed wrappers.
        max_num_param: Optional parameter budget for the primary policy.
        overflow_vram_config: Placement used after ``max_num_param``.
        **kwargs: Extra wrapper constructor arguments.

    Returns:
        The managed model with ``vram_management_enabled`` set to ``True``.

    Notes:
        ``module_map`` order matters when source classes overlap. Configure
        complete placement keys before calling; this function changes module
        identity and is normally run once during model construction.
    """
    for source_module, target_module in module_map.items():
        # If no fine-grained VRAM configuration is provided, the entire model will be managed uniformly.
        if isinstance(model, source_module):
            vram_config = fill_vram_config(model, vram_config)
            model = target_module(model, **vram_config, vram_limit=vram_limit, disk_map=disk_map, **kwargs)
            break
    else:
        enable_vram_management_recursively(
            model,
            module_map,
            vram_config,
            vram_limit=vram_limit,
            disk_map=disk_map,
            max_num_param=max_num_param,
            overflow_vram_config=overflow_vram_config,
            **kwargs,
        )
    # `vram_management_enabled` is a flag that allows the pipeline to determine whether VRAM management is enabled.
    model.vram_management_enabled = True
    return model


__all__ = [
    "AutoTorchModule",
    "AutoWrappedLinear",
    "AutoWrappedModule",
    "AutoWrappedNonRecurseModule",
    "WanAutoCastLayerNorm",
    "enable_vram_management",
    "enable_vram_management_recursively",
    "fill_vram_config",
    "move_direct_tensors_to_device",
]
