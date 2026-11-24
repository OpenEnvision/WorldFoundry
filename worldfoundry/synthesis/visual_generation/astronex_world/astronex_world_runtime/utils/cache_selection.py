"""Pick which resident history frames a causal block keeps; evicted frames are not recovered."""

import torch


def frustum_overlap(viewmats, Ks, query, candidates, depth=3.0, grid=5):
    """Fraction of the query frame's view each candidate frame saw.

    A grid of rays through the query frustum is pushed to a working depth and
    projected into each candidate; the score is the share that lands inside the
    candidate image and in front of it. Image bounds come from the principal point,
    so intrinsics may be in pixels or latents.
    """
    vm = viewmats[0].float()
    K = Ks[0, query].float() if Ks is not None else torch.eye(3, device=vm.device, dtype=vm.dtype)
    w, h = K[0, 2] * 2, K[1, 2] * 2
    ys, xs = torch.meshgrid(
        torch.linspace(0.1, 0.9, grid, device=vm.device) * h,
        torch.linspace(0.1, 0.9, grid, device=vm.device) * w,
        indexing="ij",
    )
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(grid * grid, device=vm.device)], 0)
    dirs = torch.linalg.inv(K) @ pix
    dirs = dirs / dirs.norm(dim=0, keepdim=True)

    c2w_q = torch.linalg.inv(vm[query])
    pts_w = c2w_q[:3, :3] @ (dirs * depth) + c2w_q[:3, 3:4]

    out = {}
    for c in candidates:
        pc = vm[c][:3, :3] @ pts_w + vm[c][:3, 3:4]
        uv = (Ks[0, c].float() if Ks is not None else K) @ pc
        z = pc[2]
        uv = uv[:2] / uv[2:3].clamp_min(1e-6)
        inside = (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h) & (z > 1e-3)
        out[c] = float(inside.float().mean())
    return out


def select_history_frames(body, keep_n, recent, scores, threshold, previous=(), swap=-1):
    """Return chronological resident IDs and the non-pinned subset.

    Keep recent temporal context, select other frames by overlap, and fill
    rejected slots by recency. Limit optional replacements while always filling
    the window; slots vacated by aging/pinning require mandatory replacements.
    """
    body = list(body)
    if keep_n < 0 or keep_n > len(body) or len(set(body)) != len(body):
        raise ValueError("Invalid resident history budget")
    recent = min(max(0, recent), keep_n)
    pinned = body[-recent:] if recent else []
    candidates = [f for f in body if f not in pinned]
    room = keep_n - len(pinned)
    ranked = sorted(candidates, key=lambda f: (-scores.get(f, 0.0), -f))
    selected = [f for f in ranked if scores.get(f, 0.0) >= threshold][:room]
    fallback = sorted((f for f in candidates if f not in selected), reverse=True)
    selected += fallback[: room - len(selected)]
    prev = [f for f in previous if f in candidates]
    if prev and swap >= 0:
        held = [f for f in prev if f in selected][:room]
        incoming = [f for f in selected if f not in held]
        keep_prev = [f for f in prev if f not in held]
        # New slots must be filled even if there is no previous occupant.
        budget = max(swap, room - len(prev))
        take = incoming[: min(budget, room - len(held))]
        selected = held + take
        selected += keep_prev[: room - len(selected)]
        selected += [f for f in incoming if f not in selected][: room - len(selected)]
    keep = sorted(pinned + selected)
    if len(keep) != keep_n or len(set(keep)) != keep_n:
        raise RuntimeError("History selector did not fill the window uniquely")
    return keep, selected
