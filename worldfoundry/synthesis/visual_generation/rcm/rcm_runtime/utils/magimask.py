import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.blockmask import BlockPattern, AttnMaskSpec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def make_magi_ranges_full(spec: AttnMaskSpec, Q_real: int, KV_real: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MagiAttention-style format:
      q_ranges: [R,2] int32
      k_ranges: [R,2] int32
    FULL rectangles only (union = multiple rows with identical q_range).
    """
    mode = spec.mode
    pat = spec.pattern
    assert pat is not None

    local_k = spec.local_attn_blocks - 1
    sink_k = spec.sink_blocks

    q_list: List[List[int]] = []
    k_list: List[List[int]] = []

    def _append(q0: int, q1: int, k0: int, k1: int):
        q_list.append([q0, q1])
        k_list.append([k0, k1])

    if mode == "block_causal":
        q_bo = spec.q_block_offset
        q_bounds = pat.block_bounds(q_bo=q_bo, Q=Q_real)
        n_blocks = len(q_bounds) - 1
        assert Q_real == q_bounds[-1]
        assert KV_real == pat.blocks_to_tokens(q_bo + n_blocks)

        sink_end = 0
        if sink_k > 0:
            sink_end = min(KV_real, pat.blocks_to_tokens(sink_k))

        for i in range(n_blocks):
            q0 = q_bounds[i]
            q1 = q_bounds[i + 1]

            g_blk = q_bo + i
            kv_end = min(KV_real, pat.blocks_to_tokens(g_blk + 1))

            if local_k >= 0:
                start_blk = max(0, g_blk - local_k)
                kv_start = pat.blocks_to_tokens(start_blk)
            else:
                kv_start = 0

            _append(q0, q1, kv_start, kv_end)

            if sink_end > 0 and kv_start > 0:
                _append(q0, q1, 0, min(sink_end, kv_end))

        return (
            torch.tensor(q_list, device=device, dtype=torch.int32),
            torch.tensor(k_list, device=device, dtype=torch.int32),
        )

    if mode == "teacher_forcing":
        clean_blocks = spec.clean_blocks
        assert clean_blocks > 0

        clean_len = pat.blocks_to_tokens(clean_blocks)
        total_len = 2 * clean_len

        assert Q_real == total_len and KV_real == total_len, (
            f"teacher_forcing expects Q_real==KV_real==2*clean_len ({total_len}), " f"got Q_real={Q_real}, KV_real={KV_real}"
        )

        local_k_tf = spec.local_attn_blocks - 1
        half_bounds = pat.block_bounds(q_bo=0, Q=clean_len)

        for i in range(clean_blocks):
            q0 = half_bounds[i]
            q1 = half_bounds[i + 1]

            sink_end_i = 0
            if sink_k > 0:
                sink_end_i = min(clean_len, pat.blocks_to_tokens(min(sink_k, i)))

            # ---- clean queries: kv in clean half, causal within clean ----
            kv_end = q1
            if local_k_tf >= 0:
                start_blk = max(0, i - local_k_tf)
                kv_start = half_bounds[start_blk]
            else:
                kv_start = 0

            _append(q0, q1, kv_start, kv_end)

            if sink_end_i > 0 and kv_start > 0:
                _append(q0, q1, 0, sink_end_i)

            # ---- noisy queries: shift by clean_len ----
            nq0, nq1 = q0 + clean_len, q1 + clean_len

            # (a) noisy -> previous clean blocks strictly < i  => keys in [.., q0)
            kv_end_pc = q0
            if local_k_tf >= 0:
                start_blk = max(0, i - local_k_tf)
                kv_start_pc = half_bounds[start_blk]
            else:
                kv_start_pc = 0

            if kv_end_pc > kv_start_pc:
                _append(nq0, nq1, kv_start_pc, kv_end_pc)
                if sink_end_i > 0 and kv_start_pc > 0:
                    _append(nq0, nq1, 0, sink_end_i)

            # (b) noisy -> same noisy block
            _append(nq0, nq1, nq0, nq1)

        q_ranges = torch.tensor(q_list, device=device, dtype=torch.int32)
        k_ranges = torch.tensor(k_list, device=device, dtype=torch.int32)
        return q_ranges, k_ranges

    raise ValueError(f"Unsupported mode: {mode}")


def merge_k_ranges(q_ranges: torch.Tensor, k_ranges: torch.Tensor, KV_real: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each consecutive block of identical q_range:
      - clip k to [0, KV_real]
      - drop invalid (k1<=k0)
      - sort by k0
      - merge overlaps/touching
    Asserts each q-group keeps >=1 valid k-range.
    """
    assert q_ranges.shape == k_ranges.shape and q_ranges.ndim == 2 and q_ranges.shape[1] == 2
    dev = q_ranges.device

    q_cpu = q_ranges.detach().cpu().to(torch.int64).numpy()
    k_cpu = k_ranges.detach().cpu().to(torch.int64).numpy()

    order = np.lexsort((q_cpu[:, 1], q_cpu[:, 0]))
    q_cpu = q_cpu[order]
    k_cpu = k_cpu[order]

    out_q: List[Tuple[int, int]] = []
    out_k: List[Tuple[int, int]] = []

    R = q_cpu.shape[0]
    i = 0
    while i < R:
        q0, q1 = q_cpu[i]
        assert q0 < q1, f"Invalid q range [{q0}, {q1})"
        j = i
        segs: List[Tuple[int, int]] = []
        while j < R and q_cpu[j, 0] == q0 and q_cpu[j, 1] == q1:
            k0, k1 = k_cpu[j]
            k0 = max(0, min(KV_real, int(k0)))
            k1 = max(0, min(KV_real, int(k1)))
            if k1 > k0:
                segs.append((k0, k1))
            j += 1

        assert segs, f"Empty-k group for q=[{q0},{q1})."

        segs.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = segs[0]
        for s, e in segs[1:]:
            if s <= cur_e:  # overlap or touch
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        for s, e in merged:
            out_q.append((q0, q1))
            out_k.append((s, e))

        i = j

    q2 = torch.tensor(out_q, device=dev, dtype=torch.int32)
    k2 = torch.tensor(out_k, device=dev, dtype=torch.int32)
    return q2, k2


def group_q_ranges_csr(q_ranges: torch.Tensor, k_ranges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Groups consecutive identical q_ranges into CSR:
      q_unique: [G,2] int32
      k_flat:   [R,2] int32
      qk_map:   [G+1] int32 offsets into k_flat
    Requires identical q_ranges are consecutive.
    """
    assert q_ranges.ndim == 2 and q_ranges.shape[1] == 2
    assert k_ranges.shape == q_ranges.shape

    q = q_ranges.contiguous()
    k = k_ranges.contiguous()

    same_as_prev = torch.zeros((q.shape[0],), device=q.device, dtype=torch.bool)
    if q.shape[0] > 1:
        same_as_prev[1:] = (q[1:, 0] == q[:-1, 0]) & (q[1:, 1] == q[:-1, 1])
    new_group = ~same_as_prev
    group_ids = torch.cumsum(new_group.to(torch.int32), dim=0) - 1  # [R]
    G = int(group_ids[-1].item()) + 1

    first_idx = torch.nonzero(new_group, as_tuple=False).squeeze(-1)  # [G]
    q_unique = q[first_idx].contiguous()

    counts = torch.bincount(group_ids, minlength=G)  # [G]
    qk_map = torch.zeros((G + 1,), device=q.device, dtype=torch.int32)
    qk_map[1:] = torch.cumsum(counts.to(torch.int32), dim=0)
    return q_unique, k, qk_map


def build_qtile_tasks(q_unique: torch.Tensor, Q_real: int, BLOCK_M: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build task list: each task is (group_id, qtile_id) where qtile intersects q_range.
    Returns:
      task_group: [T] int32
      task_qtile: [T] int32
    """
    device = q_unique.device
    q0 = q_unique[:, 0].to(torch.int64).clamp(0, Q_real)
    q1 = q_unique[:, 1].to(torch.int64).clamp(0, Q_real)

    t0 = (q0 // BLOCK_M).to(torch.int64)
    t1 = ((q1 + BLOCK_M - 1) // BLOCK_M).to(torch.int64)  # exclusive
    lens = (t1 - t0).clamp(min=0)

    T = int(lens.sum().item())

    groups = torch.arange(q_unique.shape[0], device=device, dtype=torch.int64)
    task_group = torch.repeat_interleave(groups, lens).to(torch.int32)

    # per-task index within its group
    prefix = torch.cumsum(lens, dim=0) - lens
    task_prefix = torch.repeat_interleave(prefix, lens)
    idx_in_group = torch.arange(T, device=device, dtype=torch.int64) - task_prefix

    task_t0 = torch.repeat_interleave(t0, lens)
    task_qtile = (task_t0 + idx_in_group).to(torch.int32)

    return task_group.contiguous(), task_qtile.contiguous()


@dataclass(frozen=True)
class MagiMask:
    # slice CSR
    q_unique: torch.Tensor  # [G,2] int32
    k_flat: torch.Tensor  # [K,2] int32
    qk_map: torch.Tensor  # [G+1] int32

    # tasks
    task_group: torch.Tensor  # [T] int32
    task_qtile: torch.Tensor  # [T] int32

    # meta
    G: int
    T: int
    BLOCK_M: int


def prepare_magi_csr_and_tasks(q_ranges: torch.Tensor, k_ranges: torch.Tensor, Q_real: int, KV_real: int, BLOCK_M: int) -> MagiMask:
    q_ranges, k_ranges = merge_k_ranges(q_ranges, k_ranges, KV_real=KV_real)
    q_unique, k_flat, qk_map = group_q_ranges_csr(q_ranges, k_ranges)
    # sanity: q coverage (partition, no gaps/overlaps)
    qu = q_unique.detach().to("cpu").to(torch.int64).numpy()
    assert qu.shape[0] > 0 and qu[0, 0] == 0 and qu[-1, 1] == Q_real, f"q_unique does not cover [0,Q_real)."
    for i in range(qu.shape[0] - 1):
        assert qu[i, 1] == qu[i + 1, 0], f"q_unique has gap/overlap at i={i}: {qu[i]} vs {qu[i+1]}"
    task_group, task_qtile = build_qtile_tasks(q_unique, Q_real=Q_real, BLOCK_M=BLOCK_M)
    return MagiMask(
        q_unique=q_unique.contiguous(),
        k_flat=k_flat.contiguous(),
        qk_map=qk_map.contiguous(),
        task_group=task_group,
        task_qtile=task_qtile,
        G=int(q_unique.shape[0]),
        T=int(task_group.numel()),
        BLOCK_M=int(BLOCK_M),
    )


def visualize_magi_ranges(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    Q: Optional[int] = None,
    K: Optional[int] = None,
    figsize: Tuple[int, int] = (7, 7),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_ranges_grid: bool = True,
):
    """
    Visualize union of Magi-style rectangles.
    """
    qr = q_ranges.detach().to("cpu").to(torch.int64).numpy()
    kr = k_ranges.detach().to("cpu").to(torch.int64).numpy()
    assert qr.shape == kr.shape and qr.shape[1] == 2

    if Q is None:
        Q = int(qr[:, 1].max(initial=0))
    if K is None:
        K = int(kr[:, 1].max(initial=0))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # draw rectangles in token coords (exact)
    ax.set_xlim(0, K)
    ax.set_ylim(Q, 0)
    for (q0, q1), (k0, k1) in zip(qr, kr):
        if q1 <= q0 or k1 <= k0:
            continue
        qq0 = max(0, min(Q, int(q0)))
        qq1 = max(0, min(Q, int(q1)))
        kk0 = max(0, min(K, int(k0)))
        kk1 = max(0, min(K, int(k1)))
        rect = Rectangle((kk0, qq0), kk1 - kk0, qq1 - qq0, fill=True, alpha=0.25, linewidth=0.8)
        ax.add_patch(rect)

    ax.set_xlabel("K (tokens)")
    ax.set_ylabel("Q (tokens)")
    ax.set_title(title or "Magi mask (rectangles)")

    if show_ranges_grid:
        # show boundaries induced by q_ranges/k_ranges (unique sorted)
        q_edges = np.unique(qr.reshape(-1))
        k_edges = np.unique(kr.reshape(-1))
        q_edges = q_edges[(q_edges >= 0) & (q_edges <= Q)]
        k_edges = k_edges[(k_edges >= 0) & (k_edges <= K)]

        # avoid too many lines
        if len(q_edges) <= 200:
            for y in q_edges:
                ax.axhline(y, linewidth=0.6, alpha=0.25)
        if len(k_edges) <= 200:
            for x in k_edges:
                ax.axvline(x, linewidth=0.6, alpha=0.25)

    fig.tight_layout()
    return fig, ax


def visualize_magi_csr_full(
    q_unique: torch.Tensor,
    k_flat: torch.Tensor,
    qk_map: torch.Tensor,
    Q: int,
    K: int,
    figsize: Tuple[int, int] = (7, 7),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    qu = q_unique.detach().to("cpu").to(torch.int64).numpy()
    kf = k_flat.detach().to("cpu").to(torch.int64).numpy()
    qm = qk_map.detach().to("cpu").to(torch.int64).numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_xlim(0, K)
    ax.set_ylim(Q, 0)  # origin upper
    ax.set_xlabel("K (tokens)")
    ax.set_ylabel("Q (tokens)")
    ax.set_title(title or "Magi FULL mask (CSR rectangles)")

    for g in range(qu.shape[0]):
        q0, q1 = qu[g]
        for idx in range(int(qm[g]), int(qm[g + 1])):
            k0, k1 = kf[idx]
            if q1 <= q0 or k1 <= k0:
                continue
            rect = Rectangle((k0, q0), k1 - k0, q1 - q0, fill=True, alpha=0.25, linewidth=0.8)
            ax.add_patch(rect)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    spec = AttnMaskSpec(
        mode="block_causal",
        pattern=BlockPattern(frame_tokens=1560, first_chunk_frames=1, chunk_frames=4),
        q_block_offset=2,
        local_attn_blocks=3,
        sink_blocks=1,
    )
    Q, KV = 1560 * 16, 1560 * 21
    q_ranges, k_ranges = make_magi_ranges_full(spec, Q_real=Q, KV_real=KV, device=torch.device("cuda"))
    print(q_ranges, k_ranges)
    fig, ax = visualize_magi_ranges(q_ranges, k_ranges, Q=Q, K=KV)
    plt.savefig("block_causal_magi.png")
    magimask = prepare_magi_csr_and_tasks(q_ranges, k_ranges, Q_real=Q, KV_real=KV, BLOCK_M=128)
    q_unique, k_flat, qk_map = magimask.q_unique, magimask.k_flat, magimask.qk_map
    print(q_unique, k_flat, qk_map)
    fig, ax = visualize_magi_csr_full(q_unique, k_flat, qk_map, Q=Q, K=KV)
    plt.savefig("block_causal_csr.png")

    tf_spec = AttnMaskSpec(
        mode="teacher_forcing",
        pattern=BlockPattern(frame_tokens=1560, first_chunk_frames=1, chunk_frames=4),
        clean_blocks=6,
        local_attn_blocks=3,
        sink_blocks=1,
    )
    Q, KV = 1560 * 21 * 2, 1560 * 21 * 2
    q_ranges, k_ranges = make_magi_ranges_full(tf_spec, Q_real=Q, KV_real=KV, device=torch.device("cuda"))
    print(q_ranges, k_ranges)
    fig, ax = visualize_magi_ranges(q_ranges, k_ranges, Q=Q, K=KV)
    plt.savefig("teacher_forcing_magi.png")
    magimask = prepare_magi_csr_and_tasks(q_ranges, k_ranges, Q_real=Q, KV_real=KV, BLOCK_M=128)
    q_unique, k_flat, qk_map = magimask.q_unique, magimask.k_flat, magimask.qk_map
    print(q_unique, k_flat, qk_map)
    fig, ax = visualize_magi_csr_full(q_unique, k_flat, qk_map, Q=Q, K=KV)
    plt.savefig("teacher_forcing_csr.png")
