# ============================================================================
# student (generator) side parallelization runtime state -- mechanism A (chunk parallel) + mechanism B (Ulysses token-SP)
#
# WARNING: this module is **fully isolated** from the teacher's `evoke_teacher/sp_runtime`:
#   only **read** it (to fetch the 8-GPU SP group size), **never write** any of its module-level globals. the
#   teacher DiT's communication params (scatter_frames / get_sp_frame_info / gather_frames) all read sp_runtime's
#   module globals, so once the student overwrote them to G_u => gather_frames would concat only G_u shards => **silent corruption in the scoring region**.
#
# two-level decomposition :
#     sp_rank = rank % G;  u_rank = sp_rank % G_u;  p_rank = sp_rank // G_u
#     G = G_p x G_u;  U-subgroup = G_u consecutive sp_ranks (same p_rank)
#   - mechanism A: chunk k's backward subgraph belongs to the slot with `p_rank == k % G_p` (**zero communication**).
#   - mechanism B: head-dim all-to-all for attention inside the U-subgroup (**pure permutation, zero reduction**).
#
# NOTE: the timing trap when reading this state (the single most important convention in this file):
#   both per-block checkpoint and section-level checkpoint re-run the forward **during backward**.
#   so "does this particular forward shard or not" must **never** be read from a mutable global inside
#   model.forward -- at recompute time the global has long left the rollout phase (GEO-REG has already run
#   too). => the caller must pass it in as an **explicit argument**, captured by the checkpoint closure. see
#   `make_ulysses_ctx()` and `EvokeTransformer3DModel.forward(sf_student_sp_ctx=...)`.
# this also replaces the `disabled()` context: instead we "**explicitly opt in at the single
#   rollout call site**", and GEO-REG / teacher / critic / eval all take the default None => cannot be sharded by accident.
# ============================================================================

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.autograd import Function

# ====== module-level state (written exactly once, by init_student_sp) ======
_inited: bool = False
_cp_enabled: bool = False          # mechanism A
_G: int = 1                        # SP group size (= sf_critic_sp_world_size), does not touch anything of the teacher's
_G_u: int = 1                      # Ulysses width (1 = mechanism B off)
_G_p: int = 1                      # = G // G_u
_sp_rank: int = 0
_u_rank: int = 0
_p_rank: int = 0
_u_group: Optional[dist.ProcessGroup] = None
_diag: bool = False
_skip_first_chunk: bool = False    # dmd_score_skip_first_chunk: g1's grad_output is structurally always 0
_seq_counter: int = 0              # collective sequence counter
_trace: bool = os.environ.get("SF_STUSP_TRACE", "0") == "1"   # trace every collective (to locate deadlocks)


class UlyssesCtx:
    """static Ulysses context for one forward (created by the caller -> captured by the checkpoint closure).

    holds only the things that "do not vary with token count"; the S-dependent shard plan is computed on the fly by model.forward (`ShardPlan`).
    """

    __slots__ = ("group", "size", "rank")

    def __init__(self, group, size: int, rank: int):
        self.group = group
        self.size = int(size)
        self.rank = int(rank)


class ShardPlan:
    """token shard plan for one forward -- **the history region and the noise region are each split into G_u parts** (dual-region split).

    NOTE: why not "one contiguous even split of the whole sequence" (the v1 approach, rejected by measurement):
      the sequence is `[history(b) | noise(ocl)]`, and `ocl < b` always holds (540/2160/8640 vs 2400) => under a contiguous even split
      **the first few ranks get history only, with an empty noise slice** => they must skip attn2 (an empty tensor blows up flash)
      => **two ranks in the group create different numbers of autograd nodes => sequence_nr shifts wholesale => the engine's
      interleaving order over the several independent branches (per-chunk subgraphs + the GEO-REG graph) can differ => all-to-all issue order shifts => deadlock**.
      J1, measured twice on separate runs: every rank with u_rank=1 hung on the U-subgroup's
      ALLTOALL_BASE, with a different backlog count per subgroup (41/33/41/41, data dependent). turning off section-level recompute did not help
      -- the interleaving order is a **property of the graph**, unrelated to recompute.
    => the dual-region split makes **every rank always hold both history and noise** => attn2 is never empty => the skip branch disappears
      => graphs are strictly isomorphic within the group. two side benefits: (1) **zero pad** at the real numbers (b=2400, ocl=540/2160/8640, G_u=2
      -> lh=1200, ln=270/1080/4320, all divide evenly); (2) load becomes even (previously stage0's attn2 all piled onto the last rank).

    local layout = `[hist_u (lh) | noise_u (ln)]`, equal length `L = lh + ln` on every rank (all-to-all requires equal length):
      lh = ceil(b/G_u),  ln = ceil(ocl/G_u);  rank u holds H[u*lh:(u+1)*lh] and N[u*ln:(u+1)*ln]
      => `hist_local == lh` always holds for every rank (it no longer depends on u_rank).
    the "global" order attention sees is `[h_0|n_0|h_1|n_1|...]` -- a **permutation** of the true order. dense mask-free
    attention is invariant to key/value order, and RoPE is applied per token **before** the all-to-all => legal.
    pad can only land at the tail of each region (only when b or ocl is not divisible by G_u, i.e. stage0 of the warp section) => it is
    dropped exactly by `valid_index()`, not by assuming "pad is always at the very end".
    """

    __slots__ = ("ctx", "S_real", "hist_global", "noise_global", "lh", "ln", "L", "_vidx")

    def __init__(self, ctx: UlyssesCtx, S_real: int, hist_global: int):
        g = ctx.size
        self.ctx = ctx
        self.S_real = int(S_real)
        self.hist_global = int(hist_global)
        self.noise_global = int(S_real) - int(hist_global)
        assert self.noise_global >= g, (
        f"[STU-SP] noise token count {self.noise_global} < G_u={g}: the dual-region split requires at least 1 noise token per rank")
        assert self.hist_global >= 0
        self.lh = (self.hist_global + g - 1) // g
        self.ln = (self.noise_global + g - 1) // g
        self.L = self.lh + self.ln
        self._vidx = None

    @property
    def hist_local(self) -> int:
        """the history/noise split point inside this rank's shard -- equals lh on every rank under the dual-region split."""
        return self.lh

    @property
    def has_pad(self) -> bool:
        return (self.lh * self.ctx.size != self.hist_global) or (self.ln * self.ctx.size != self.noise_global)

    def valid_index(self, device) -> torch.Tensor:
        """indices of the **real tokens** in the permuted global sequence `[h_0|n_0|h_1|n_1|...]` (length G_u*L).

        position p = u*L + j:  j < lh -> global history element u*lh+j (real iff < b);
                           j >= lh -> global noise   element u*ln+(j-lh) (real iff < ocl).
        only used when `has_pad` (index_select before flash, scatter back on the way out). cached on the plan.
        """
        if self._vidx is None:
            g, lh, ln, L = self.ctx.size, self.lh, self.ln, self.L
            keep = []
            for u in range(g):
                for j in range(lh):
                    if u * lh + j < self.hist_global:
                        keep.append(u * L + j)
                for j in range(ln):
                    if u * ln + j < self.noise_global:
                        keep.append(u * L + lh + j)
            self._vidx = torch.tensor(keep, dtype=torch.long)
        return self._vidx.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------
def init_student_sp(
    sp_world_size: int,
    chunk_parallel: bool,
    ulysses_size: int = 1,
    diag: bool = False,
    skip_first_chunk: bool = False,
):
    """build the U-subgroups. **every rank must call this in the same order** (dist.new_group is a collective).

: `new_group(timeout=None)` falls back to the 10-minute default (it does not inherit init_process_group's 1800s)
            => timeout must be passed explicitly, same source as sp_runtime.py (NCCL_TIMEOUT, in milliseconds).
: when G_u == world_size, reuse WORLD (do not build a subgroup, avoids crossing with the DeepSpeed PG);
            when G_u == 1, mechanism B is off and no group is needed.
    """
    global _inited, _cp_enabled, _G, _G_u, _G_p, _sp_rank, _u_rank, _p_rank
    global _u_group, _diag, _skip_first_chunk, _seq_counter

    import datetime

    _cp_enabled = bool(chunk_parallel)
    _G_u = int(ulysses_size)
    _diag = bool(diag)
    _skip_first_chunk = bool(skip_first_chunk)
    _seq_counter = 0
    _u_group = None

    if not _cp_enabled and _G_u <= 1:
        # both mechanisms off => keep the off path bit-identical, do nothing
        _G, _G_p, _G_u = 1, 1, 1
        _sp_rank = _u_rank = _p_rank = 0
        _inited = True
        return

    assert dist.is_available() and dist.is_initialized(), "[STU-SP] requires an already initialized torch.distributed"
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    _G = int(sp_world_size)
    assert _G >= 1 and world_size % _G == 0, f"[STU-SP] world_size={world_size} is not divisible by G={_G}"
    assert _G % _G_u == 0, f"[STU-SP] G={_G} is not divisible by G_u={_G_u}"
    _G_p = _G // _G_u
    _sp_rank = rank % _G
    _u_rank = _sp_rank % _G_u
    _p_rank = _sp_rank // _G_u

    if _G_u > 1:
        _timeout = datetime.timedelta(milliseconds=int(os.environ.get("NCCL_TIMEOUT", "1800000")))
        if _G_u == world_size:
            # subgroup == WORLD => reuse it, do not create a new one (same reason as sp_runtime.py)
            _u_group = dist.group.WORLD
        else:
            # every rank iterates all U-subgroups in the same order: by SP group first, then by p_rank within the group
            for base in range(0, world_size, _G):
                for p in range(_G_p):
                    ranks = [base + p * _G_u + u for u in range(_G_u)]
                    grp = dist.new_group(ranks, timeout=_timeout)
                    if rank in ranks:
                        _u_group = grp
            assert _u_group is not None, f"[STU-SP] rank={rank} did not land in any U-subgroup"

    _inited = True
    if rank % _G == 0:
        print(
            f"[STU-SP] init: world={world_size} G={_G} -> G_p={_G_p} x G_u={_G_u}; "
            f"chunk_parallel={_cp_enabled} ulysses={_G_u > 1} diag={_diag} "
            f"skip_first_chunk={_skip_first_chunk} (rank {rank}: p_rank={_p_rank} u_rank={_u_rank})",
            flush=True,
        )


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------
def is_cp_enabled() -> bool:
    return _cp_enabled


def is_ulysses_enabled() -> bool:
    return _G_u > 1


def is_any_enabled() -> bool:
    return _cp_enabled or _G_u > 1


def get_G() -> int:
    return _G


def get_G_p() -> int:
    return _G_p


def get_G_u() -> int:
    return _G_u


def get_p_rank() -> int:
    return _p_rank


def get_u_rank() -> int:
    return _u_rank


def get_u_group():
    return _u_group


def is_diag() -> bool:
    return _diag


def loss_scale() -> int:
    """: under mechanism A/B the 8 ranks hold **non-overlapping partial sums** of the same clip's gradient => the WORLD average is short by a factor G => loss x G.

    isomorphic to the critic's `_c_loss_scale = 1 if decouple else sp_world_size` (train_evoke.py).
    load imbalance (5 sections vs 4) does not matter: the DMD loss denominator is `mask.sum()`, independent of how many sections this rank holds.
    """
    return _G if is_any_enabled() else 1


def redundant_grad_div() -> int:
    """: an extra `/G_u` for mechanism B's "redundant parameter set".

    the sharded region = the 40 blocks. trainable params **outside** the sharded region (upstream patch_embedding/patch_short/mid/long/
    condition_embedder, downstream proj_out/norm_out) receive a **redundant full copy** inside the U-subgroup instead of a partial sum
    (the entry scatter's backward is an all_gather => every rank reconstructs the full-length gradient) => after the uniform xG they are G_u times too large
    => the effective learning rate is silently inflated while **the loss still converges** = the most insidious class of bug here.
    """
    return _G_u if is_ulysses_enabled() else 1


def make_ulysses_ctx() -> Optional[UlyssesCtx]:
    """created at the rollout call site (the only opt-in point). mechanism B off -> None => the transformer takes the original path."""
    if _G_u <= 1 or _u_group is None:
        return None
    return UlyssesCtx(_u_group, _G_u, _u_rank)


# ---------------------------------------------------------------------------
# mechanism A: ownership function (, reproducible within the group, no data dependence)
# ---------------------------------------------------------------------------
def cp_owns(k: int) -> bool:
    """whether this rank builds the backward subgraph of chunk k. always True when off => old behaviour bit-identical.

    round-robin is hard-coded (simplification: no snake knob, load differs by <=1 section ~= +3%).
    under `dmd_score_skip_first_chunk`, g1 (k==0) has a structurally always-zero grad_output
    (the boolean mask's backward scatters exact 0 into unselected positions) => **always** skip building the graph for that whole section, saving 4.9s.
    """
    if not _cp_enabled:
        return True
    if _skip_first_chunk and k == 0:
        return False
    return (int(k) % _G_p) == _p_rank


def set_skip_first_chunk(v: bool) -> None:
    """per-step toggle for "structurally skip g1's backward or not".

    motivation: on an i2v step the teacher's I-frame slot is the real reference image, so g1's DMD gradient is no longer always 0 (gradient_mask covers it) =>
    mechanism A must build the graph for g1, otherwise "masked in but nobody computes the gradient" = silently dropping 1/20 of the supervision. on a v2v step, switch back to True.

    WARNING: the caller must guarantee **the same value within the SP group** (otherwise the ranks in a group get different chunk ownership sets -> the sum of section gradients is not the gradient of any single loss).
    mechanism A's chunk ownership (cp_owns / G_p = SP group size) is itself an **intra-group** notion, so "consistent within the group" is sufficient.
    WARNING: do not phrase it as "consistent across all ranks" -- that is neither achievable nor needed:
      - image-only samples are always i2v, and the sampler's shard_key=ith_sp_group => **same sample within a group** (intra-group consistent, ok),
        but **different groups may get different samples** => group 0 may be an i2v image while group 1 is a v2v video.
      - when video samples are drawn by ratio the seed carries _i2v_grp => that too is a **per-group** value, not one value shared by all ranks.
    divergence between groups is not a problem: there is no collective inside this branch, and both modes traverse the same set of student params => DeepSpeed's
    world-level reduction stays symmetric. seeding uses a local random.Random instance and does not consume the global RNG => intra-group noise symmetry is preserved.
    only called when sf_i2v_score_g1=true; otherwise _skip_first_chunk keeps its init value => old behaviour bit-identical.
    only called when sf_i2v_score_g1=true; otherwise _skip_first_chunk keeps its init value => old behaviour bit-identical.
    """
    global _skip_first_chunk
    _skip_first_chunk = bool(v)


def get_skip_first_chunk() -> bool:
    return _skip_first_chunk


# ---------------------------------------------------------------------------
# mechanism B: communication primitives (, pure permutation, zero reduction)
# ---------------------------------------------------------------------------
def _diag_device():
    """where diagnostic tensors live. CPU/gloo unit tests have no cuda => fall back to cpu (the diagnostic logic itself is device agnostic)."""
    return torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")


def bump_seq(n: int = 1, op: str = "", shape=None):
    """sequence counter: the graphs of two ranks in a U-subgroup are bit-for-bit isomorphic => the counts must agree, and the first misalignment errors out immediately.

    with `SF_STUSP_TRACE=1` it additionally traces every call (per-rank stderr). a deadlock leaves **no post-mortem scene**, so trace is the only handle:
    diff the traces of two ranks in the group and the first divergence is the misalignment point. the trace carries the **thread name** -- main thread = normal forward,
    `pt_autograd_*` = the checkpoint recompute inside backward => the three phases forward/recompute/backward are told apart at a glance.
    """
    global _seq_counter
    _seq_counter += int(n)
    if _trace:
        import threading
        try:
            r = dist.get_rank()
        except Exception:
            r = -1
        print(f"[STU-SP-TRACE r{r} u{_u_rank} #{_seq_counter} {op} {tuple(shape) if shape is not None else ''} "
              f"thr={threading.current_thread().name}", flush=True)


def get_seq() -> int:
    return _seq_counter


def check_seq_in_group(tag: str = ""):
    """compare the sequence counter across the U-subgroup via all_reduce(min/max). only called under diag, once on the first step."""
    if _u_group is None:
        return
    t = torch.tensor([_seq_counter, -_seq_counter], dtype=torch.long, device=_diag_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=_u_group)
    hi, lo = int(t[0].item()), -int(t[1].item())
    assert hi == lo == _seq_counter, (
        f"[STU-SP SEQ{(' ' + tag) if tag else ''}] collective sequence numbers disagree inside the U-subgroup: "
        f"local={_seq_counter} min={lo} max={hi} => forward graphs are not isomorphic within the group (recompute timing / branch misalignment), will deadlock or compute wrong gradients"
    )


_selfcheck: bool = os.environ.get("SF_STUSP_SELFCHECK", "0") == "1"
_selfcheck_done: bool = False


def selfcheck_pending() -> bool:
    """`SF_STUSP_SELFCHECK=1` and not done yet => True. default False, zero overhead."""
    return _selfcheck and not _selfcheck_done


def selfcheck_report(got: torch.Tensor, ref: torch.Tensor, plan, tag: str = ""):
    """single-forward "sharded vs unsharded" comparison -- pulls "is the implementation correct" out of "bf16 autoregressive compounding".

    mechanism B changes the head partition and the GEMM shapes => every forward necessarily differs at the bf16 level (~1e-3); and the rollout autoregression
    (N chunks x 3 stages x 40 blocks in series) amplifies that to the percent level => **"compare the whole rollout's loss
    bit-for-bit against unsharded" is structurally invalid for mechanism B** (J1d: dmd_loss_lw 0.269 vs 0.285). only the single-forward relative difference is a criterion for implementation correctness:
      <~1e-2 => bf16 noise, implementation correct (a floor set by dtype, not a bug)
      O(1)   => genuinely wrong (token misalignment / unpaired permutation / pad leakage)
    """
    global _selfcheck_done
    _selfcheck_done = True
    d = (got.detach().float() - ref.detach().float()).abs()
    scale = ref.detach().float().abs().max().clamp_min(1e-12)
    b = plan.hist_global
    print(f"[STU-SP SELFCHECK {tag}] single forward, sharded vs unsharded: max|delta|={d.max().item():.3e} "
          f"rel={d.max().item() / scale.item():.3e} (history region {d[:, :b].max().item():.3e} / "
          f"noise region {d[:, b:].max().item():.3e}); dtype={ref.dtype} "
          f"lh={plan.lh} ln={plan.ln} L={plan.L} pad={plan.has_pad} "
          f"=> {'bf16 noise magnitude, implementation correct' if d.max().item() / scale.item() < 3e-2 else '!! magnitude too large, implementation has a bug'}",
          flush=True)


def selfcheck_grad(run_fn, full_inputs, shard_inputs, plan, params, tag: str = "",
                   depths=(1, 5, 10, 20), n_layers: int = 40):
    """NOTE: gradient-equivalence selfcheck: **the sharded gradient summed over the U-subgroup** vs **the unsharded gradient**, same cotangent, single forward.

    under the "bf16 + autoregression" constraints this is the only criterion that can really answer "are the gradients equivalent":
      - the whole rollout cannot be compared -- mechanism B necessarily changes GEMM shapes / head partition => ~1e-3 per forward (bf16 mantissa 8 bit),
        and 12 forwards compound autoregressively to the percent level => however correct the implementation, it can never pass "bit-for-bit / 1e-6".
      - but **a single forward + the same cotangent** has no compounding: if the sharding is correct, then in-block params are
        **partial sums** inside the U-subgroup, and `sum_u g_u` must equal the unsharded `g` (to bf16 precision). this verifies both that backward is the transpose of forward,
        and the very "partial sum" semantics that the xG scaling relies on.
      - criterion: relative L2 <~1e-2 (the bf16 floor) => equivalent; O(1), or cos clearly off 1 => genuinely wrong.

    `run_fn(hidden, tsp, rot, sp_plan) -> out`; the sharded path's out must already be gathered back to full length (same shape as the reference).
    use a fixed cotangent (a deterministic tensor determined by the shape, not random, so both paths get exactly the same one).
    """
    global _selfcheck_done
    _selfcheck_done = True

    def _one(upto):
        """compare once at a given depth. **the depth sweep is the discriminator between "dtype accumulation" and "structural bug"**:
        dtype noise grows monotonically with depth (depth=1 should be ~= bf16 eps 4e-3); a structural bug is already large at depth=1."""
        with torch.enable_grad():
            ref_out = run_fn(*(t.detach() for t in full_inputs), None, upto)
            # NOTE: the cotangent must be a **full-rank random direction**. it used to be linspace(-1,1,numel).to(bf16): nearly constant along channel,
            #   varying extremely slowly along token (tens of millions of elements squeezed into a few hundred bf16 values) => a very badly conditioned probe direction, insensitive to token misalignment,
            #   and it also amplified benign bf16 rounding into false alarms (pointed out in review). switched to seeded randn + an intra-group consistency assertion.
            _g = torch.Generator(device="cpu").manual_seed(20260725)
            cot = torch.randn(ref_out.shape, generator=_g, dtype=torch.float32).to(ref_out.device)
            assert_same_in_group(int(cot.sum().mul(1e6).item()), "cotangent fingerprint", plan.ctx.group)
            g_ref = torch.autograd.grad((ref_out.float() * cot.float()).sum(), params,
                                        allow_unused=True, retain_graph=False)
            fwd_ref = ref_out.detach().float()
            sh_out = run_fn(*(t.detach() for t in shard_inputs), plan, upto)
            g_sh = torch.autograd.grad((sh_out.float() * cot.float()).sum(), params,
                                       allow_unused=True, retain_graph=False)
            fwd_rel = ((sh_out.detach().float() - fwd_ref).norm()
                       / fwd_ref.norm().clamp_min(1e-30)).item()
        n_ok = n_bad = n_none = n_tiny = 0
        worst = (0.0, "", 0.0, 0.0)
        dot = nr = ns = 0.0
        for (name, _), a, b in zip(params_named(params), g_ref, g_sh):
            if a is None or b is None:
                n_none += 1
                continue
            bs = b.detach().float().clone()
            dist.all_reduce(bs, group=plan.ctx.group)      # NOTE: partial sums => only the intra-group sum equals the reference
            af = a.detach().float()
            na, dn = af.norm().item(), (bs - af).norm().item()
            rel = dn / max(na, 1e-30)
            dot += float((af * bs).sum()); nr += float((af * af).sum()); ns += float((bs * bs).sum())
            # NOTE: when the reference itself is near 0 the relative error blows up into a false alarm => count it separately, keep it out of the verdict
            if na < 1e-6:
                n_tiny += 1
            elif rel <= 3e-2:
                n_ok += 1
            else:
                n_bad += 1
            if rel > worst[0] and na >= 1e-6:
                worst = (rel, name, na, dn)
        cos = dot / max((nr ** 0.5) * (ns ** 0.5), 1e-30)
        print(f"[STU-SP SELFCHECK-GRAD {tag} depth={upto:>3}] fwd_rel={fwd_rel:.3e} | "
              f"ok={n_ok} bad={n_bad} tiny(|ref|<1e-6)={n_tiny} none={n_none} cos={cos:.8f} "
              f"worst_rel={worst[0]:.3e}@{worst[1]} (|ref|={worst[2]:.3e} |delta|={worst[3]:.3e})", flush=True)
        return cos, n_bad

    # depth sweep: 1 -> full depth. already bad at depth=1 => structural bug; monotone growth with depth => dtype accumulation.
    for d in depths:
        if d <= n_layers:
            _one(d)
    cos, n_bad = _one(n_layers)
    verdict = "gradients equivalent (within bf16 precision)" if (n_bad == 0 and cos > 0.999) else "!! not equivalent (read the depth sweep to characterize it)"
    print(f"[STU-SP SELFCHECK-GRAD {tag}] verdict: {verdict}", flush=True)


_PARAM_NAMES = []


def set_param_names(named):
    global _PARAM_NAMES
    _PARAM_NAMES = list(named)


def params_named(params):
    if len(_PARAM_NAMES) == len(params):
        return _PARAM_NAMES
    return [(f"p{i}", p) for i, p in enumerate(params)]


def check_ipg_bucket_order(engine, tag: str = "") -> int:
    """: **the param insertion order inside the IPG bucket must be identical across all of WORLD** -- if it is not, gradients are silently scrambled.

    mechanism (deepspeed 0.14.5 `stage_1_and_2.py`): `reduce_ipg_grads` -> `average_tensor` accumulates `curr_size` along
    the **insertion order** of `params_in_ipg_bucket` to derive each slice's `(dst, bucket_offset, numel)`
    (`:1043-1144`), and `ipg_buffer` is filled in that same insertion order. a rank's insertion order = the execution order of each param's
    `AccumulateGrad` in the backward graph. mechanisms A/B make each rank's backward graph **structurally different** (5 sections vs 4; inside the U-subgroup
    u=0 skips attn2 at stage0/1 while u=1 does not) => order consistency is **not proven**. once it differs, the same offset range holds
    a different param on different ranks, and allreduce adds them together => the whole bucket's gradients are scrambled, and **separating the reductions helps not at all here**.

    why it is most likely safe (but must still be verified) -- WARNING, correction: the justification written here was originally **backwards**, do not copy it again.
      the wrong old claim: "AccumulateGrad's sequence_nr is assigned on iteration 0 and is far below any node of this iteration => it gets pushed to
        after the backward graph has drained". the truth is the opposite: torch explicitly gives them **the highest priority**
        (`torch/csrc/autograd/function.h`: "we prioritize AccumulateGrad nodes by explicitly setting
        its sequence_nr to be UINT64_MAX") => as soon as one is ready it jumps the queue and runs, it is never deferred.
      the real justification: each param's AccumulateGrad runs exactly once per backward, and it waits for **all** incoming edges to arrive
        (dependency count down to zero) => its position is decided by "the last contribution to arrive". the engine drains in descending sequence_nr => the last
        arriving contribution always comes from the **earliest constructed** retained-graph forward, i.e. the earliest retained
        stage of the chunk with the **smallest k** owned by this rank. which chunk / which stage differs per rank, but each is **one complete forward of the same module**
        => the backward arrival order (proj_out -> blocks 39..0 -> patch_embedding / patch_short|mid|long /
        condition_embedder) is rank independent. that is the actual reason bucket order is consistent under mechanism A.
    the conclusion is unchanged (still safe, still must be backstopped by the assertion), but the reasoning is replaced -- a wrong justification is more dangerous than none.

    if it really is inconsistent, the fix already prepared = a **gradient-order anchor**: at the very start of the step (before the rollout) build
    `anchor = sum_p (p.view(-1)[0] * 0)` and add it to the loss. its nodes have the lowest `sequence_nr` => processed last in backward
    => every param's **last** contribution comes from the anchor, so the order = the anchor's construction order = the `named_parameters()`
    order => bit-identical across all ranks, and what is added is 0.0 (loss/gradients bit-unchanged).

    returns this rank's hash (handy for printing). must be called **before** `allreduce_gradients()` (after it the bucket is empty).

    WARNING WARNING coverage (re-checked, do not read a PASS as "the full order has been verified"): in this config **the bucket necessarily overflows** --
      generator trainable numel = 625,721,344 vs reduce_bucket_size = 2e8 in zero2_dualteacher_gen_sp.json
      => within one backward `reduce_ipg_grads()` flushes at least 3 times, and every flush clears
      `params_in_ipg_bucket` (stage_1_and_2.py) => this function is called **after** backward, so what it sees is only the **residual bucket**
      left after the last flush (~=25.7M elements, about 4% of the parameters). **the order of the earlier buckets has never been verified.**
      more importantly: in the overflow regime this check sits **downstream** of the failure it wants to catch -- if the insertion orders really differ per rank, the flush boundaries
      and slice shapes differ with them => mismatched NCCL collectives, hanging inside backward, so this point is never reached at all.
      what it can still catch is the "same-numel permutation within the residual bucket" class (which is exactly the silent gradient scrambling), so it stays.
      NOTE: also, if the last param happens to trigger a flush, seq==[] => hash = crc32("") on all ranks => the assertion passes trivially.
        **seeing hash=0 in the log means this step verified nothing**, do not take it for a pass.
      to really cover the full order: raise reduce_bucket_size to >= 625,721,344 (with overlap_comm:false that costs only one extra
      ipg_buffer), or hook `reduce_ipg_grads` to accumulate a hash per bucket.
    """
    import zlib

    opt = getattr(engine, "optimizer", None)
    bucket = getattr(opt, "params_in_ipg_bucket", None)
    if bucket is None:
        print("WARN: optimizer has no params_in_ipg_bucket (not ZeRO-1/2?), skipping the bucket-order check", flush=True)
        return -1
    # param_id is generated from the traversal order of self.bit16_groups => identical on all ranks for the same model => directly comparable across ranks
    seq = [int(e[2]) for e in bucket]
    h = zlib.crc32((",".join(map(str, seq))).encode()) & 0x7FFFFFFF
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([h, -h, len(seq), -len(seq)], dtype=torch.long, device=_diag_device())
        dist.all_reduce(t, op=dist.ReduceOp.MAX)      # WORLD: that is exactly the communication domain of the DP reduction
        hi, lo, nhi, nlo = int(t[0]), -int(t[1]), int(t[2]), -int(t[3])
        assert hi == lo == h and nhi == nlo == len(seq), (
            f"[STU-SP §8-(6){(' ' + tag) if tag else ''}] param order inside the IPG bucket **differs across ranks**: "
            f"local hash={h} (min={lo} max={hi}), local len={len(seq)} (min={nlo} max={nhi}) => "
            f"average_tensor will add the gradients of different params together (silent scrambling). for the fix see "
            f"the 'gradient-order anchor' in this function's docstring.")
    return h


def assert_same_in_group(value: int, name: str, group=None):
    """assert that an integer quantity is consistent within the group (the S check of). unequal => fail-fast, otherwise all-to-all shapes mismatch -> hang."""
    grp = group if group is not None else _u_group
    if grp is None:
        return
    v = int(value)
    t = torch.tensor([v, -v], dtype=torch.long, device=_diag_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=grp)
    hi, lo = int(t[0].item()), -int(t[1].item())
    assert hi == lo == v, f"[STU-SP] {name} inconsistent within the group: local={v} min={lo} max={hi}"


def _seq_to_head(x: torch.Tensor, group, gu: int) -> torch.Tensor:
    """[B, s, H, D] --all_to_all--> [B, gu*s, H//gu, D]. pure permutation."""
    B, s, H, D = x.shape
    hl = H // gu
    t = x.view(B, s, gu, hl, D).permute(2, 0, 1, 3, 4).contiguous()   # [gu, B, s, hl, D] dim0=destination rank (head group)
    o = torch.empty_like(t)
    dist.all_to_all_single(o, t, group=group)                          # o[j] = rank j's token slice x this rank's head group
    bump_seq(op="a2a_s2h", shape=x.shape)
    return o.permute(1, 0, 2, 3, 4).reshape(B, gu * s, hl, D)


def _head_to_seq(y: torch.Tensor, group, gu: int) -> torch.Tensor:
    """[B, gu*s, H//gu, D] --all_to_all--> [B, s, H, D]. the exact inverse of `_seq_to_head`."""
    B, S, hl, D = y.shape
    s = S // gu
    t = y.view(B, gu, s, hl, D).permute(1, 0, 2, 3, 4).contiguous()   # [gu, B, s, hl, D] dim0=destination rank (token slice)
    o = torch.empty_like(t)
    dist.all_to_all_single(o, t, group=group)                          # o[j] = this rank's token slice x rank j's head group
    bump_seq(op="a2a_h2s", shape=y.shape)
    return o.permute(1, 2, 0, 3, 4).reshape(B, s, gu * hl, D)


class _A2ASeqToHead(Function):
    @staticmethod
    def forward(ctx, x, group, gu):
        ctx.group, ctx.gu = group, gu
        return _seq_to_head(x, group, gu)

    @staticmethod
    def backward(ctx, grad):
        return _head_to_seq(grad.contiguous(), ctx.group, ctx.gu), None, None


class _A2AHeadToSeq(Function):
    @staticmethod
    def forward(ctx, x, group, gu):
        ctx.group, ctx.gu = group, gu
        return _head_to_seq(x, group, gu)

    @staticmethod
    def backward(ctx, grad):
        return _seq_to_head(grad.contiguous(), ctx.group, ctx.gu), None, None


def a2a_seq_to_head(x: torch.Tensor, ctx: UlyssesCtx) -> torch.Tensor:
    return _A2ASeqToHead.apply(x, ctx.group, ctx.size)


def a2a_head_to_seq(x: torch.Tensor, ctx: UlyssesCtx) -> torch.Tensor:
    return _A2AHeadToSeq.apply(x, ctx.group, ctx.size)


def _all_gather_tokens(x_local: torch.Tensor, group, gu: int) -> torch.Tensor:
    """[B, L, *rest] --all_gather--> [B, gu*L, *rest], concatenated in rank order (= the global token order)."""
    xt = x_local.transpose(0, 1).contiguous()                          # [L, B, *rest]
    out = torch.empty((gu * xt.shape[0],) + tuple(xt.shape[1:]), dtype=xt.dtype, device=xt.device)
    dist.all_gather_into_tensor(out, xt, group=group)
    bump_seq(op="allgather", shape=x_local.shape)
    return out.transpose(0, 1).contiguous()


def _pad_tokens(x: torch.Tensor, n: int) -> torch.Tensor:
    """zero-pad the tail of dim=1 up to length n. under the dual-region split the padding is done **per region** (once for history, once for noise),
    so pad is no longer "always at the very end of the sequence" -- dropping it must go through `ShardPlan.valid_index()`."""
    cur = x.shape[1]
    if cur == n:
        return x
    assert cur < n, f"[STU-SP] _pad_tokens: current length {cur} > target {n}"
    return torch.cat([x, x.new_zeros((x.shape[0], n - cur) + tuple(x.shape[2:]))], dim=1)


def _split_local(full: torch.Tensor, plan: ShardPlan) -> torch.Tensor:
    """[B, S_real, *] (layout [H(b)|N(ocl)]) -> this rank's [B, L, *] = [h_u(lh) | n_u(ln)], zero communication."""
    b, u, lh, ln = plan.hist_global, plan.ctx.rank, plan.lh, plan.ln
    h = _pad_tokens(full[:, :b], plan.lh * plan.ctx.size)[:, u * lh : (u + 1) * lh]
    n = _pad_tokens(full[:, b:], plan.ln * plan.ctx.size)[:, u * ln : (u + 1) * ln]
    return torch.cat([h, n], dim=1).contiguous()


def _merge_global(gathered: torch.Tensor, plan: ShardPlan) -> torch.Tensor:
    """[B, G_u*L, *] (the rank-concatenated permuted order [h_0|n_0|h_1|n_1|...]) -> [B, S_real, *] in true order."""
    g, lh, ln, L = plan.ctx.size, plan.lh, plan.ln, plan.L
    hs = [gathered[:, u * L : u * L + lh] for u in range(g)]
    ns = [gathered[:, u * L + lh : (u + 1) * L] for u in range(g)]
    return torch.cat([torch.cat(hs, dim=1)[:, : plan.hist_global],
                      torch.cat(ns, dim=1)[:, : plan.noise_global]], dim=1)


class _ScatterTokens(Function):
    """entry: full length -> this rank's dual-region shard. forward has zero communication; backward = all_gather + inverse permutation, **then /G_u**.

    NOTE: that `/G_u` is the structural fix made after two independent reviews on, replacing the earlier heuristic of
    "decide by module tree + param hook" (which had two fatal bugs):
      - the all_gather in backward makes **every rank reconstruct the full-length gradient** => upstream params (patch_embedding / patch_short/mid/long /
        condition_embedder.time_embedder / time_proj) are a **redundant full copy** inside the U-subgroup, so the uniform xG makes them G_u times too large.
      - dividing **here**, instead of hooking the params, buys two critical things:
        (1) it only acts on gradients that **crossed this shard boundary** => the `text_embedder` side path (which enters attn2 via
           the unsharded `encoder_hidden_states` and is itself a token partial sum) is naturally not divided -- a param hook cannot tell them apart and would certainly halve it wrongly;
        (2) it only acts on **the forward that was actually sharded** => GEO-REG (sp_plan=None, does not cross this boundary, and sits in the same
           backward as DMD) is naturally not divided -- a param hook sees the sum of the two terms and is **expressively incapable** of separating them.
    """

    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        return _split_local(x, plan)

    @staticmethod
    def backward(ctx, grad):
        p = ctx.plan
        full = _merge_global(_all_gather_tokens(grad.contiguous(), p.ctx.group, p.ctx.size), p)
        return (full / p.ctx.size) if p.ctx.size > 1 else full, None


class _ScaleGrad(Function):
    """forward = identity, backward = xk. used to divide the **parameter** gradients of the "tail after gather"
    (norm_out / proj_out / temb modulation) by G_u while leaving the gradient flowing back into the block region **unchanged**:
        y = gather(...);  y = ScaleGrad(y, G_u);  out = tail(y, temb, ...);  out = ScaleGrad(out, 1/G_u)
    algebra: the cotangent entering the tail is v/G_u => dL/dtheta_tail picks up 1/G_u (ok); dL/dy picks up 1/G_u and is then restored by the preceding xG_u (ok).
    together with the /G_u in `_ScatterTokens.backward`, this covers both classes of redundant full-copy params: "after gather" and "before scatter".
    """

    @staticmethod
    def forward(ctx, x, k):
        ctx.k = float(k)
        return x

    @staticmethod
    def backward(ctx, grad):
        return (grad * ctx.k) if ctx.k != 1.0 else grad, None


def scale_grad(x: torch.Tensor, k: float) -> torch.Tensor:
    return _ScaleGrad.apply(x, k) if k != 1.0 else x


class _GatherTokens(Function):
    """exit: this rank's dual-region shard -> full-length true order. backward = local slicing + zero fill
    => the pad region's gradient is always 0 => a pad token's contribution to any weight gradient is **exactly 0** (no varlen mask needed)."""

    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        return _merge_global(_all_gather_tokens(x, plan.ctx.group, plan.ctx.size), plan)

    @staticmethod
    def backward(ctx, grad):
        return _split_local(grad.contiguous(), ctx.plan), None


def scatter_tokens(x: torch.Tensor, plan: ShardPlan) -> torch.Tensor:
    return _ScatterTokens.apply(x, plan)


def gather_tokens(x: torch.Tensor, plan: ShardPlan) -> torch.Tensor:
    return _GatherTokens.apply(x, plan)


def drop_pad_tokens(x: torch.Tensor, plan: ShardPlan) -> torch.Tensor:
    """drop pad tokens before attention: [B, G_u*L, H, D] -> [B, S_real, H, D].
    uses `index_select` (autograd built in; backward scatters the gradient back into place and fills pad positions with 0) => pad contributes exactly 0 to weight gradients."""
    return x.index_select(1, plan.valid_index(x.device))


def restore_pad_tokens(x: torch.Tensor, plan: ShardPlan) -> torch.Tensor:
    """after attention, put zeros back at the pad positions (so the head->seq all-to-all stays equal length)."""
    idx = plan.valid_index(x.device)
    out = x.new_zeros((x.shape[0], plan.L * plan.ctx.size) + tuple(x.shape[2:]))
    return out.index_copy(1, idx, x)


def phase_barrier():
    """: explicit device-side sync at the "U-subgroup phase -> 8-GPU SP group phase" boundary.

    official torch warning (distributed_c10d.py): concurrent use of multiple NCCL PGs is unsafe; a collective on one PG
    must be **complete on the device** (not merely enqueued) before a collective on another PG can be enqueued. logical time-slicing is not enough, this sync is needed.
    """
    if _u_group is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def install_redundant_grad_hooks(model, log: bool = True) -> int:
    """DEPRECATED (, after two independent reviews).

    the original implementation hooked `g/G_u` onto params based on "is it under `model.blocks`", which had **two fatal bugs**:
      (1) `condition_embedder.text_embedder` lives outside blocks, yet its gradient **is a token partial sum**
         (it enters attn2 via the unsharded `encoder_hidden_states`, while the ranks' query sets are disjoint)
         => it was wrongly divided down to 0.5x, **halving the effective lr while the loss curve looks perfectly normal**;
      (2) a param-level hook sees the **total gradient** `(G*q_p + w*h)/G_u`, so the GEO-REG term (redundant on all ranks, should be x1)
        gets divided too -- DMD and GEO-REG are in the **same** backward, arbitrarily interleaved by the engine, so any phase switch is useless
        => **a param-level hook is expressively incapable** of distinguishing loss terms.
    it is now done at the two shard boundaries in the graph: the /G_u in `_ScatterTokens.backward` (covers the upstream params before the scatter)
    + the `_ScaleGrad` pair around the tail (covers the proj_out/norm_out/temb path after the gather). both act only on
    **the path that crossed a shard boundary** => text_embedder and GEO-REG are naturally unharmed, and no name/module heuristic is relied on any more.
    """
    raise AssertionError(
        "[STU-SP] install_redundant_grad_hooks is deprecated: /G_u is now done in _ScatterTokens.backward and in the tail's "
        "_ScaleGrad pair (acting only on the sharded path, so text_embedder and GEO-REG are naturally unharmed). see this function's docstring.")
