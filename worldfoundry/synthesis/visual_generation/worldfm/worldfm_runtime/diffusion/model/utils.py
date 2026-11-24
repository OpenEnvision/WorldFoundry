import os
import random
import re
import sys
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torchvision import transforms as T

from worldfoundry.core.nn.blocks.tuples import ntuple as _ntuple

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if isinstance(module, Iterable):
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, **kwargs)
        else:
            # torch.utils.checkpoint.checkpoint does NOT accept **kwargs in some torch versions.
            # We wrap the module call in a closure so checkpoint only sees positional args.
            if kwargs:

                def _forward_with_kwargs(*inputs):
                    return module(*inputs, **kwargs)

                try:
                    # Newer torch versions recommend setting this explicitly.
                    return checkpoint(_forward_with_kwargs, *args, use_reentrant=get_checkpoint_use_reentrant())
                except TypeError:
                    return checkpoint(_forward_with_kwargs, *args)
            try:
                return checkpoint(module, *args, use_reentrant=get_checkpoint_use_reentrant())
            except TypeError:
                return checkpoint(module, *args)
    return module(*args, **kwargs)


# -----------------------------------------------------------------------------
# Checkpoint behavior toggle
# -----------------------------------------------------------------------------
# `use_reentrant` impacts checkpoint implementation details and can affect
# stability/performance depending on torch version and model.
#
# Default: False (current behavior in this repo after recent changes).
_CHECKPOINT_USE_REENTRANT: bool = False


def set_checkpoint_use_reentrant(flag: bool) -> None:
    global _CHECKPOINT_USE_REENTRANT
    _CHECKPOINT_USE_REENTRANT = bool(flag)


def get_checkpoint_use_reentrant() -> bool:
    return _CHECKPOINT_USE_REENTRANT


def checkpoint_sequential(functions, step, input, *args, **kwargs):  # noqa: F811 - Later definition replaces the earlier one, as upstream.
    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input, *args)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # the last chunk has to be non-volatile
    end = -1
    segment = len(functions) // step
    for start in range(0, step * (segment - 1), step):
        end = start + step - 1
        input = checkpoint(run_function(start, end, functions), input, preserve_rng_state=preserve)
    return run_function(end + 1, len(functions) - 1, functions)(input)


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, tensor.ndim)))


#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################
def get_mask(batch, length, mask_ratio, device, mask_type=None, data_info=None, extra_len=0):
    """
    Get the binary mask for the input sequence.
    Args:
        - batch: batch size
        - length: sequence length
        - mask_ratio: ratio of tokens to mask
        - data_info: dictionary with info for reconstruction
    return:
        mask_dict with following keys:
        - mask: binary mask, 0 is keep, 1 is remove
        - ids_keep: indices of tokens to keep
        - ids_restore: indices to restore the original order
    """
    assert mask_type in ["random", "fft", "laplacian", "group"]
    mask = torch.ones([batch, length], device=device)
    len_keep = int(length * (1 - mask_ratio)) - extra_len

    if mask_type == "random" or mask_type == "group":
        noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_removed = ids_shuffle[:, len_keep:]

    elif mask_type in ["fft", "laplacian"]:
        if "strength" in data_info:
            strength = data_info["strength"]

        else:
            N = data_info["N"][0]
            img = data_info["ori_img"]
            _, C, H, W = img.shape
            if mask_type == "fft":
                # Split the image into N x N patches: (batch, C, H/N, N, W/N, N)
                reshaped_image = img.reshape((batch, -1, H // N, N, W // N, N))
                fft_image = torch.fft.fftn(reshaped_image, dim=(3, 5))
                # Frequency strength per patch: sum of FFT magnitudes
                strength = torch.sum(torch.abs(fft_image), dim=(1, 3, 5)).reshape(
                    (
                        batch,
                        -1,
                    )
                )
            elif type == "laplacian":  # noqa: E721 - Upstream compares against the builtin `type` object.
                laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).reshape(
                    1, 1, 3, 3
                )
                laplacian_kernel = laplacian_kernel.repeat(C, 1, 1, 1)
                # Split the image into N x N patches: (-1, C, N, N)
                reshaped_image = img.reshape(-1, C, H // N, N, W // N, N).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, N, N)
                laplacian_response = F.conv2d(reshaped_image, laplacian_kernel, padding=1, groups=C)
                strength = laplacian_response.sum(dim=[1, 2, 3]).reshape(
                    (
                        batch,
                        -1,
                    )
                )

        # Sample the token order with probabilities proportional to the normalized strength.
        probabilities = strength / (strength.max(dim=1)[0][:, None] + 1e-5)
        ids_shuffle = torch.multinomial(probabilities.clip(1e-5, 1), length, replacement=False)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_removed = ids_shuffle[:, len_keep:]

    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return {"mask": mask, "ids_keep": ids_keep, "ids_restore": ids_restore, "ids_removed": ids_removed}


def mask_out_token(x, ids_keep, ids_removed=None):
    """
    Mask out the tokens specified by ids_keep.
    Args:
        - x: input sequence, [N, L, D]
        - ids_keep: indices of tokens to keep
    return:
        - x_masked: masked sequence
    """
    N, L, D = x.shape  # batch, length, dim
    x_remain = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    if ids_removed is not None:
        x_masked = torch.gather(x, dim=1, index=ids_removed.unsqueeze(-1).repeat(1, 1, D))
        return x_remain, x_masked
    else:
        return x_remain


def mask_tokens(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def unmask_tokens(x, ids_restore, mask_token):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+1, D]
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    x = torch.cat([x, mask_tokens], dim=1)
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    return x


# Parse 'None' to None and others to float value
def parse_float_none(s):
    assert isinstance(s, str)
    return None if s == "None" else float(s)


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def init_processes(fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = str(random.randint(2000, 6000))
    print(f"MASTER_ADDR = {os.environ['MASTER_ADDR']}")
    print(f"MASTER_PORT = {os.environ['MASTER_PORT']}")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=args.global_rank, world_size=args.global_size)
    fn(args)
    if args.global_size > 1:
        cleanup()


def mprint(*args, **kwargs):
    """
    Print only from rank 0.
    """
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def cleanup():
    """
    End DDP training.
    """
    dist.barrier()
    mprint("Done!")
    dist.barrier()
    dist.destroy_process_group()


# ----------------------------------------------------------------------------
# logging info.
class Logger:
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name=None, file_mode="w", should_flush=True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)  # noqa: SIM115 - Keeps the upstream file-handle lifetime.

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, text):
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self):
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self):
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def prepare_prompt_ar(prompt, ratios, device="cpu", show=True):
    # get aspect_ratio or ar
    aspect_ratios = re.findall(r"--aspect_ratio\s+(\d+:\d+)", prompt)
    ars = re.findall(r"--ar\s+(\d+:\d+)", prompt)
    custom_hw = re.findall(r"--hw\s+(\d+:\d+)", prompt)
    if show:
        print("aspect_ratios:", aspect_ratios, "ars:", ars, "hws:", custom_hw)
    prompt_clean = prompt.split("--aspect_ratio")[0].split("--ar")[0].split("--hw")[0]
    if len(aspect_ratios) + len(ars) + len(custom_hw) == 0 and show:
        print(
            "Wrong prompt format. Set to default ar: 1. change your prompt into format '--ar h:w or --hw h:w' for correct generating"
        )
    if len(aspect_ratios) != 0:
        ar = float(aspect_ratios[0].split(":")[0]) / float(aspect_ratios[0].split(":")[1])
    elif len(ars) != 0:
        ar = float(ars[0].split(":")[0]) / float(ars[0].split(":")[1])
    else:
        ar = 1.0
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    if len(custom_hw) != 0:
        custom_hw = [float(custom_hw[0].split(":")[0]), float(custom_hw[0].split(":")[1])]
    else:
        custom_hw = ratios[closest_ratio]
    default_hw = ratios[closest_ratio]
    prompt_show = f"prompt: {prompt_clean.strip()}\nSize: --ar {closest_ratio}, --bin hw {ratios[closest_ratio]}, --custom hw {custom_hw}"
    return (
        prompt_clean,
        prompt_show,
        torch.tensor(default_hw, device=device)[None],
        torch.tensor([float(closest_ratio)], device=device)[None],
        torch.tensor(custom_hw, device=device)[None],
    )


def resize_and_crop_tensor(samples: torch.Tensor, new_width: int, new_height: int):
    orig_hw = torch.tensor([samples.shape[2], samples.shape[3]], dtype=torch.int)
    custom_hw = torch.tensor([int(new_height), int(new_width)], dtype=torch.int)

    if (orig_hw != custom_hw).all():
        ratio = max(custom_hw[0] / orig_hw[0], custom_hw[1] / orig_hw[1])
        resized_width = int(orig_hw[1] * ratio)
        resized_height = int(orig_hw[0] * ratio)

        transform = T.Compose([T.Resize((resized_height, resized_width)), T.CenterCrop(custom_hw.tolist())])
        return transform(samples)
    else:
        return samples


def resize_and_crop_img(img: Image, new_width, new_height):
    orig_width, orig_height = img.size

    ratio = max(new_width / orig_width, new_height / orig_height)
    resized_width = int(orig_width * ratio)
    resized_height = int(orig_height * ratio)

    img = img.resize((resized_width, resized_height), Image.LANCZOS)

    left = (resized_width - new_width) / 2
    top = (resized_height - new_height) / 2
    right = (resized_width + new_width) / 2
    bottom = (resized_height + new_height) / 2

    img = img.crop((left, top, right, bottom))

    return img


def mask_feature(emb, mask):
    if emb.shape[0] == 1:
        keep_index = mask.sum().item()
        return emb[:, :, :keep_index, :], keep_index
    else:
        masked_feature = emb * mask[:, None, :, None]
        return masked_feature, emb.shape[2]
