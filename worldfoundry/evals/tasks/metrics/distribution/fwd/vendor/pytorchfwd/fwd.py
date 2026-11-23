"""Frechet Wavelet Distance computation."""

import os
import pathlib

import numpy as np
import torch as th
import torchvision.transforms as tv
from tqdm import tqdm

from .freq_math import calculate_frechet_distance, forward_wavelet_packet_transform
from .utils import ImagePathDataset, _parse_args

th.set_default_dtype(th.float64)


IMAGE_EXTS = {"jpg", "jpeg", "png"}
NUM_PROCESSES = None


def compute_packet_statistics(
    dataloader: th.utils.data.DataLoader, wavelet: str, max_level: int, log_scale: bool
) -> tuple[np.ndarray, ...]:
    """Compute wavelet packet transform across batches.

    Args:
        dataloader (th.utils.data.DataLoader): Torch dataloader.
        wavelet (str): Choice of wavelet.
        max_level (int): Wavelet decomposition level.
        log_scale (bool): Apply log scale.

    Returns:
        Tuple[np.ndarray, ...]: Mean and sigma for each packet.
    """
    packets = []
    device = th.device("cuda:0") if th.cuda.is_available() else th.device("cpu")
    for img_batch in tqdm(dataloader):
        if isinstance(img_batch, list):
            img_batch = img_batch[0]
        img_batch = img_batch.to(device)
        packets.append(forward_wavelet_packet_transform(img_batch, wavelet, max_level, log_scale).cpu())
    packet_tensor = th.cat(packets, dim=0)
    packet_tensor = th.permute(packet_tensor, (1, 0, 2, 3, 4))
    P, BS, C, H, W = packet_tensor.shape
    packet_tensor = th.reshape(packet_tensor, (P, BS, C * H * W))
    print("Computing mean and std for each packet.")
    mu = th.mean(packet_tensor, dim=1).numpy()

    def gpu_cov(tensor_):
        return th.cov(tensor_.T).cpu()

    sigma = th.stack([gpu_cov(packet_tensor[p, :, :].to(device)) for p in range(P)], dim=0).numpy()
    return mu, sigma


def calculate_path_statistics(
    path: str,
    wavelet: str,
    max_level: int,
    log_scale: bool,
    batch_size: int,
    resize: int | None,
) -> tuple[np.ndarray, ...]:
    """Compute mean and sigma for given path.

    Args:
        path (str): npz path or image directory.
        wavelet (str): Choice of wavelet.
        max_level (int): Decomposition level.
        log_scale (bool): Apply log scale.
        batch_size (int): Batch size for packet decomposition.
        resize (Union[int, None]): Optional resize option.

    Raises:
        ValueError: Error if mu and sigma cannot be calculated.

    Returns:
        Tuple[np.ndarray, ...]: Tuple containing mean and sigma for each packet.
    """
    mu, sigma = None, None
    if path.endswith((".npz", ".npy")):
        with np.load(path) as fp:
            mu = fp["mu"][:]
            sigma = fp["sigma"][:]
    else:
        posfix_path = pathlib.Path(path)
        img_names = sorted([name for ext in IMAGE_EXTS for name in posfix_path.glob(f"*.{ext}")])
        transfs_list = []
        if resize is not None:
            print(f"Resizing images to {(resize, resize)} resolution")
            transfs_list.append(tv.Resize((resize, resize)))
        transfs_list.append(tv.ToTensor())
        dataloader = th.utils.data.DataLoader(
            ImagePathDataset(img_names, transforms=tv.Compose(transfs_list)),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_PROCESSES,
        )
        mu, sigma = compute_packet_statistics(
            dataloader=dataloader,
            wavelet=wavelet,
            max_level=max_level,
            log_scale=log_scale,
        )

    if (mu is None) or (sigma is None):
        raise ValueError(f"The file path: {path} is empty/doesn't have statistics.")
    return mu, sigma


def _compute_avg_frechet_distance(mu1, mu2, sigma1, sigma2):
    """Compute avg frechet distance over packets."""
    frechet_distances = []
    for packet_no in tqdm(range(len(mu1))):
        fd = calculate_frechet_distance(
            mu1=mu1[packet_no, :],
            mu2=mu2[packet_no, :],
            sigma1=sigma1[packet_no, :, :],
            sigma2=sigma2[packet_no, :, :],
        )
        frechet_distances.append(fd)
    return np.mean(frechet_distances)


def compute_fwd(
    paths: list[str],
    wavelet: str,
    max_level: int,
    log_scale: bool,
    batch_size: int,
    resize: int | None,
) -> float:
    """Compute Frechet Wavelet Distance.

    Args:
        paths (List[str]): List containing path of source and generated images.
        wavelet (str): Choice of wavelet.
        max_level (int): Decomposition level.
        log_scale (bool): Apply log scale.
        batch_size (int): Batch size for packet decomposition.
        resize (Union[int, None]): Optional resize option.

    Raises:
        RuntimeError: Error if path doesn't exist.

    Returns:
        float: Frechet Wavelet Distance.
    """
    for path in paths:
        if not os.path.exists(path):
            raise RuntimeError(f"Invalid path: {path}")

    print(f"Computing stats for path: {paths[0]}")
    mu_1, sigma_1 = calculate_path_statistics(paths[0], wavelet, max_level, log_scale, batch_size, resize)
    print(f"Computing stats for path: {paths[1]}")
    mu_2, sigma_2 = calculate_path_statistics(paths[1], wavelet, max_level, log_scale, batch_size, resize)

    print("Computing Frechet distances for each packet.")
    return _compute_avg_frechet_distance(mu_1, mu_2, sigma_1, sigma_2)


def _save_packets(
    paths: list[str],
    wavelet: str,
    max_level: int,
    log_scale: bool,
    batch_size: int,
    resize: int | None,
) -> None:
    """
    Args:
        paths (List[str]): List of paths containing input and output files.
        wavelet (str): Choice of wavelet.
        max_level (int): Decomposition level.
        log_scale (bool): Apply log scale.
        batch_size (int): Batch size for packet decomposition.
        resize (Union[int, None]): Optional resize option.

    Raises:
        RuntimeError: Error if input path is invalid.
        RuntimeError: Error if the output file already exists.
    """
    if not os.path.exists(paths[0]):
        raise RuntimeError(f"Invalid path: {paths[0]}")

    if os.path.exists(paths[1]):
        raise RuntimeError(f"Stats file already exists at the given path: {paths[1]}")

    print(f"Computing stats for path: {paths[0]}")
    mu_1, sigma_1 = calculate_path_statistics(paths[0], wavelet, max_level, log_scale, batch_size, resize)
    np.savez_compressed(paths[1], mu=mu_1, sigma=sigma_1)


def main():
    """Compute FWD given paths."""
    global NUM_PROCESSES

    th.manual_seed(0)
    args = _parse_args()
    print(args)
    if args.num_processes is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()
        NUM_PROCESSES = min(num_cpus, 16) if num_cpus is not None else 0
    else:
        NUM_PROCESSES = args.num_processes
    print(f"Num work: {NUM_PROCESSES}")
    if args.deterministic:
        th.use_deterministic_algorithms(True)
    if args.save_packets:
        _save_packets(
            args.path,
            args.wavelet,
            args.max_level,
            args.log_scale,
            args.batch_size,
            args.resize,
        )
        return

    fwd = compute_fwd(
        args.path,
        args.wavelet,
        args.max_level,
        args.log_scale,
        args.batch_size,
        args.resize,
    )
    print(f"FWD: {fwd}")


if __name__ == "__main__":
    main()
