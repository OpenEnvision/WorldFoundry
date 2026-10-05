"""RocketScienceDataset: load time-aligned 4-perspective clips of Rocket League matches.

One WebDataset sample is one *(match, chunk)*: a ~4 s window (80 frames at 20 fps) of a match,
bundling all perspectives — members `p{i}.mp4` / `p{i}.jsonl` (ordered by `player_id`) + `meta.json`,
sample key `{match_id}_c{chunk:05d}`. Each chunk carries its own per-frame action and physics track.
A clip is taken from within a single chunk.

Two access patterns:
  * `load_match(match_id, ...)` — random access (reads only the chunks the requested clips need)
  * `iter_clips(...)`           — stream shards, one decoded clip at a time (each sample = one chunk)
"""

from __future__ import annotations

import json
import logging
import random
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .actions import KeyVocab, tensorize_actions
from .clips import compute_clip_frame_indices, compute_stride
from .decode import decode_frames
from .events import Event, events_in_frame_window, overlaps_any, parse_anchors, replay_spans
from .schema import Index, MatchEntry
from .state import FrameState

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

N_PLAYERS = 4


def _read_member(tar: tarfile.TarFile, member: "tarfile.TarInfo | str") -> bytes:
    """Read a tar member's bytes, raising if it is missing or not a regular file."""
    info = tar.getmember(member) if isinstance(member, str) else member
    f = tar.extractfile(info)
    if f is None:
        raise ValueError(f"tar member is not a readable file: {info.name}")
    return f.read()


def _parse_member(name: str) -> tuple[str, str]:
    """Split a tar member basename into (sample_key, field). The WebDataset key is everything before
    the first dot (the chunk key has no dots); the field is the rest, e.g. `p0.mp4` / `meta.json`."""
    key, _, field = name.partition(".")
    return key, field


def _perspective_index(field: str) -> int | None:
    """Perspective index from a field like `p0.mp4`/`p3.jsonl`, or None for non-`p{i}` fields."""
    if not field.startswith("p") or "." not in field:
        return None
    head = field.split(".", 1)[0][1:]
    return int(head) if head.isdigit() else None


def chunk_key(match_id: str, chunk_idx: int) -> str:
    """WebDataset sample key for one (match, chunk).

    The WebDataset key is everything before the first '.', so `match_id` must contain no '.' (else
    member parsing would split the key wrongly).
    """
    if "." in match_id:
        raise ValueError(f"match_id must not contain '.': {match_id!r}")
    return f"{match_id}_c{chunk_idx:05d}"


def parse_chunk_key(key: str) -> tuple[str, int] | None:
    """Inverse of `chunk_key`: `{match_id}_c{idx}` -> (match_id, idx), or None if not a chunk key."""
    base, sep, idx = key.rpartition("_c")
    if not sep or not base or not idx.isdigit():
        return None
    return base, int(idx)


@dataclass
class MatchClip:
    """One time-aligned within-chunk window of the selected perspectives of a match.

    The perspective axis (size P) is ordered by `player_id` (NOT by team — read `teams`). With the
    default `perspective="all"`, P=4; selecting a single perspective gives P=1.

    `frame_indices` are **chunk-local** source-frame indices (into chunk `chunk_idx`), shared across
    perspectives. `clip_id` is a per-match running index over all chunks' clips (chunk 0's clips
    first, then chunk 1's, ...). Full per-perspective metadata is in `metadata`.
    """

    match_id: str
    chunk_idx: int
    clip_id: int
    perspectives: list[int]  # selected source-perspective indices (0..n_players-1), player_id order
    frame_indices: list[int]  # chunk-local source-frame indices, shared across perspectives
    stride: int
    target_fps: int
    src_fps: float  # the match's true source fps (frames / duration); exact, not target_fps * stride
    player_ids: list[int]  # per selected perspective
    teams: list[int]  # per selected perspective
    recording_offsets: list[float]  # recording_offset_sec per selected perspective (master clock)
    metadata: list[dict]  # full per-perspective metadata
    actions: "torch.Tensor"  # (P, T, n_keys) int32 multi-hot
    events: list[Event]  # events overlapping this window (mapped via the first selected perspective)
    frames: "torch.Tensor | None" = None  # (P, T, C, H, W) uint8; None if decode=False
    physics: list[list[FrameState]] | None = None  # per perspective, T per-frame game-state dicts
    # (None if the dataset carries no physics). All perspectives share the same world state; kept
    # per-perspective so it stays frame-aligned with that perspective's frames/actions.
    global_frame_indices: list[int] | None = None  # match-global source-frame index per step (same T
    # as frame_indices). Lets a step be placed on the master clock — master_sec = recording_offset +
    # g / src_fps — for event/phase mapping that must see anchors outside the clip window.
    video_bytes: list[bytes] | None = None  # per selected perspective, the chunk's compressed mp4
    # bytes (set when decode=False and carry_video=True), for decoding a drawn clip later via
    # `decode_perspective`. Shared by reference across all clips read from the same chunk.

    def decode_perspective(self, p: int, frame_size: tuple[int, int] | None = None) -> "torch.Tensor":
        """Decode selected perspective `p`'s carried video bytes to a (T, C, H, W) uint8 tensor."""
        if self.video_bytes is None:
            raise ValueError("clip carries no video_bytes; read it with carry_video=True")
        return decode_frames(self.video_bytes[p], self.frame_indices, frame_size)


@dataclass
class _ClipPlan:
    """A single planned within-chunk clip (computed from the index, before any bytes are read)."""

    clip_id: int
    chunk_idx: int
    local: list[int]  # chunk-local source-frame indices
    g0: int  # match-global start frame (for event/replay mapping)
    g_end: int  # match-global end frame (exclusive)


@dataclass
class _MatchPlan:
    """Per-match constants: the full within-chunk clip plan + alignment metadata."""

    entry: MatchEntry
    plan: list[_ClipPlan]
    stride: int
    src_fps: float
    events: list[Event]
    meta: list[dict]
    replays: list[tuple[int, int]]


class RocketScienceDataset:
    def __init__(self, index_path: str | Path, vocab: KeyVocab | None = None):
        self.index_path = Path(index_path)
        self.root = self.index_path.parent
        self.vocab = vocab or KeyVocab.default_rl()

        self.index: Index = Index.load(self.index_path)
        self.matches: dict[str, MatchEntry] = {}
        # original-chunk-index -> position in chunk_frames, per match (chunks may be non-contiguous)
        self._chunk_pos: dict[str, dict[int, int]] = {}
        for e in self.index.entries:
            e.perspectives.sort(key=lambda p: p.player_id)  # define the p0..p3 axis
            self.matches[e.match_id] = e
            self._chunk_pos[e.match_id] = {e.chunk_id(pos): pos for pos in range(len(e.chunk_frames))}

    @classmethod
    def from_local(cls, path: str | Path, **kwargs) -> "RocketScienceDataset":
        """`path` may be the dataset directory (containing index.json) or the index.json itself."""
        path = Path(path)
        if path.is_dir():
            path = path / "index.json"
        if not path.exists():
            raise FileNotFoundError(f"No index at {path}.")
        return cls(path, **kwargs)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        split: str | None = None,
        subdir: str | None = None,
        revision: str | None = None,
        shards: int | None = None,
        **kwargs,
    ) -> "RocketScienceDataset":
        """Load one split from a WorldFoundry-local Hugging Face snapshot.

        The repo holds each split under its own prefix: `{split}/index.json` plus WebDataset shards,
        flat (`test/dataset_*.tar`) or in numbered subfolders (`train/000/dataset_*.tar`);
        `index.json` records the shard paths, so both layouts load identically. Pass `split` to
        choose one. `subdir` overrides the prefix for a non-split layout. ``repo_id`` may also be
        an explicit local snapshot directory. This in-tree runtime never contacts the network;
        materialize the dataset with the project checkpoint tooling before inference.

        Pass `shards=N` to restrict the dataset to matches in the first `N` local tar shards.
        ``revision`` is accepted for upstream API compatibility; immutable revision selection is
        performed when the local snapshot is materialized.

        The tars are standard WebDataset shards and can also be read with
        `datasets.load_dataset("webdataset", data_files="hf://datasets/<repo>/train/*/*.tar", ...)`
        (flat splits: `.../test/*.tar`).
        """
        from worldfoundry.core.io.paths import resolve_local_hf_model_path

        prefix = subdir if subdir is not None else (split if split is not None else "data")
        del revision
        local = resolve_local_hf_model_path(
            repo_id,
            required_files=(f"{prefix}/index.json",),
        )

        if shards is None:
            return cls.from_local(Path(local) / prefix, **kwargs)

        if shards < 1:
            raise ValueError(f"shards must be >= 1 (or None for all), got {shards}")
        index = Index.load(Path(local) / prefix / "index.json")
        keep_shards = sorted({e.shard for e in index.entries})[:shards]
        missing_shards = [name for name in keep_shards if not (Path(local) / prefix / name).is_file()]
        if missing_shards:
            raise FileNotFoundError(
                "MIRA's local Rocket Science snapshot is incomplete; missing shards: "
                + ", ".join(missing_shards)
            )
        ds = cls.from_local(Path(local) / prefix, **kwargs)
        ds._restrict_to_shards(set(keep_shards))
        return ds

    def _restrict_to_shards(self, keep_shards: set[str]) -> None:
        """Drop every match whose shard is not in `keep_shards` (used by `from_hub(shards=...)`)."""
        self.index.entries = [e for e in self.index.entries if e.shard in keep_shards]
        self.index.total_samples = len(self.index.entries)
        self.matches = {mid: e for mid, e in self.matches.items() if e.shard in keep_shards}
        self._chunk_pos = {mid: p for mid, p in self._chunk_pos.items() if mid in self.matches}

    def match_ids(self) -> list[str]:
        return list(self.matches)

    # -- per-match planning (no bytes read) -------------------------------

    def max_clip_frames(self, target_fps: int) -> int:
        """Longest ``clip_len`` (in ``target_fps`` frames) that fits inside a single chunk of at
        least one match, or ``0`` if the dataset holds no usable match.

        A clip of ``clip_len`` indices spans ``(clip_len - 1) * stride + 1`` source frames and must
        fit within one chunk (see :meth:`_plan_match`), so per match the largest feasible ``clip_len``
        is ``(max(chunk_frames) - 1) // stride + 1``. Matches whose fps is not an integer multiple of
        ``target_fps`` (``compute_stride`` raises) are skipped, mirroring the streaming read path.
        This reads only index metadata (frame counts / durations), no video.
        """
        best = 0
        for entry in self.matches.values():
            if not entry.chunk_frames or not entry.perspectives:
                continue
            duration = entry.perspectives[0].duration
            if duration <= 0:
                continue
            src_fps = sum(entry.chunk_frames) / duration
            try:
                stride = compute_stride(src_fps, target_fps)
            except ValueError:
                continue
            best = max(best, (max(entry.chunk_frames) - 1) // stride + 1)
        return best

    def _plan_match(self, entry: MatchEntry, clip_len: int, target_fps: int) -> _MatchPlan:
        """Enumerate every within-chunk clip of a match; a clip whose span exceeds the largest
        chunk raises."""
        persp = entry.perspectives
        total = sum(entry.chunk_frames)
        for p in persp:
            if p.frames != total:
                raise ValueError(
                    f"{entry.match_id}: perspective {p.player_id} has {p.frames} frames != "
                    f"sum(chunk_frames)={total}"
                )
        src_fps = total / persp[0].duration
        stride = compute_stride(src_fps, target_fps)

        # A clip of clip_len indices at this stride spans (clip_len-1)*stride + 1 source frames.
        span = (clip_len - 1) * stride + 1
        biggest = max(entry.chunk_frames)
        if span > biggest:
            raise ValueError(
                f"{entry.match_id}: a clip_len={clip_len} clip @ {target_fps}fps spans {span} source "
                f"frames, but a clip must fit in one chunk (largest is {biggest} frames). "
                f"Reduce clip_len or raise target_fps."
            )

        offsets = [0]
        for f in entry.chunk_frames:
            offsets.append(offsets[-1] + f)

        plan: list[_ClipPlan] = []
        cid = 0
        for c, n_frames in enumerate(entry.chunk_frames):
            chunk_clips, _ = compute_clip_frame_indices(n_frames, src_fps, clip_len, target_fps)
            for local in chunk_clips:
                g0 = offsets[c] + local[0]
                g_end = offsets[c] + local[-1] + stride
                plan.append(_ClipPlan(cid, c, local, g0, g_end))
                cid += 1

        events_all = parse_anchors(persp[0].anchors)  # anchors identical across perspectives
        replays = replay_spans(events_all, src_fps, persp[0].recording_offset_sec, total)
        return _MatchPlan(entry, plan, stride, src_fps, events_all, [p.model_dump() for p in persp], replays)

    @staticmethod
    def _select_plan(
        mp: _MatchPlan, exclude_replays: bool, clip_ids: list[int] | None, max_clips: int | None
    ) -> list[_ClipPlan]:
        """Apply clip_ids / exclude_replays / max_clips to a match's clip plan. `clip_ids` selects by
        the per-match running `clip_id`."""
        want = set(clip_ids) if clip_ids is not None else None
        out: list[_ClipPlan] = []
        for cp in mp.plan:
            if want is not None and cp.clip_id not in want:
                continue
            if exclude_replays and overlaps_any(cp.g0, cp.g_end, mp.replays):
                continue
            out.append(cp)
            if max_clips is not None and len(out) >= max_clips:
                break
        return out

    @staticmethod
    def _select_perspectives(perspective: str | int, n: int, rng: "random.Random") -> list[int]:
        """Resolve the `perspective` selector to a list of source-perspective indices.

        "all" -> [0..n-1]; "random" -> one random index (per call, so per clip when streaming);
        an int i -> [i]; "playerK" (1-based) -> [K-1].
        """
        if isinstance(perspective, int):
            if not 0 <= perspective < n:
                raise ValueError(f"perspective index {perspective} out of range [0,{n})")
            return [perspective]
        if perspective == "all":
            return list(range(n))
        if perspective == "random":
            return [rng.randrange(n)]
        if isinstance(perspective, str) and perspective.startswith("player"):
            k = int(perspective.removeprefix("player")) - 1
            if not 0 <= k < n:
                raise ValueError(f"{perspective!r} out of range (have {n} players)")
            return [k]
        raise ValueError(f"Invalid perspective {perspective!r}; use 'all', 'random', 'playerK', or int")

    # -- assembling one within-chunk clip --------------------------------

    def _assemble(
        self, mp: _MatchPlan, cp: _ClipPlan, chunk: list[dict], sel: list[int],
        target_fps: int, decode: bool, frame_size, action_fps: int | None = None,
        carry_video: bool = False,
    ) -> MatchClip:  # fmt: skip
        """Build a MatchClip for `cp` from a single chunk's bytes. `chunk[i]` is
        `{"video": bytes, "lines": list[bytes]}` for perspective i.

        `action_fps` decouples the action sample rate from the frame `target_fps`. When `None`
        (default) actions are tensorized at `target_fps` — one action step per decoded frame. When
        set, actions are tensorized at `action_fps` over the same clip line-span, so the action time
        dim becomes `clip_len * (action_fps // target_fps)` (e.g. 10fps frames + 20fps actions -> 2
        actions/frame). `action_fps` must be a positive integer multiple of `target_fps`.
        """
        import torch

        persp = mp.entry.perspectives
        clip_len = len(cp.local)
        l0, l_end = cp.local[0], cp.local[-1] + mp.stride

        if action_fps is None:
            act_fps, n_action_steps = target_fps, clip_len
        else:
            if action_fps < target_fps or action_fps % target_fps != 0:
                raise ValueError(
                    f"action_fps ({action_fps}) must be a positive integer multiple of "
                    f"target_fps ({target_fps})"
                )
            act_fps, n_action_steps = action_fps, clip_len * (action_fps // target_fps)

        per_actions = []
        for i in sel:
            # All perspectives are frame-aligned, so actions use the same planned stride as the
            # frame indices (mp.src_fps), not a per-perspective re-derivation. `keep_last_partial`
            # OR-s a short final window (a clip running to the chunk's end) into a real step rather
            # than dropping it, so the trailing step carries the keys actually held over its frames.
            a = tensorize_actions(
                chunk[i]["lines"][l0:l_end], self.vocab, mp.src_fps, act_fps, keep_last_partial=True
            )
            # A clip ending at the chunk boundary can have a short final action window (the chunk
            # ends mid-window), so with action_fps > target_fps (multiple action steps per frame) it
            # yields fewer than `n_action_steps`. Hold the last step to pad back to the nominal count
            # so actions stay aligned to the clip's frames. (No-op when the window is full, e.g.
            # action_fps == target_fps, which is every released config.)
            if a.shape[0] < n_action_steps:
                a = torch.cat([a, a[-1:].expand(n_action_steps - a.shape[0], -1)], dim=0)
            if a.shape[0] != n_action_steps:
                raise ValueError(
                    f"{mp.entry.match_id} clip {cp.clip_id}: {a.shape[0]} action steps != {n_action_steps}"
                )
            per_actions.append(a)
        actions = torch.stack(per_actions, dim=0)

        frames = None
        if decode:
            frames = torch.stack([decode_frames(chunk[i]["video"], cp.local, frame_size) for i in sel], dim=0)
            if frames.shape[1] != clip_len:
                raise ValueError(
                    f"{mp.entry.match_id} clip {cp.clip_id}: decoded {frames.shape[1]} != {clip_len}"
                )

        physics = None
        if all(chunk[i].get("physics") is not None for i in sel):
            physics = [[json.loads(chunk[i]["physics"][f]) for f in cp.local] for i in sel]

        offs = [p.recording_offset_sec for p in persp]
        return MatchClip(
            match_id=mp.entry.match_id,
            chunk_idx=cp.chunk_idx,
            clip_id=cp.clip_id,
            perspectives=list(sel),
            frame_indices=cp.local,
            stride=mp.stride,
            target_fps=target_fps,
            src_fps=mp.src_fps,
            player_ids=[persp[i].player_id for i in sel],
            teams=[persp[i].team for i in sel],
            recording_offsets=[offs[i] for i in sel],
            metadata=[mp.meta[i] for i in sel],
            actions=actions,
            events=events_in_frame_window(mp.events, cp.g0, cp.g_end, mp.src_fps, offs[sel[0]]),
            frames=frames,
            physics=physics,
            global_frame_indices=[cp.g0 + (li - cp.local[0]) for li in cp.local],
            video_bytes=[chunk[i]["video"] for i in sel] if carry_video else None,
        )

    # -- reading chunk samples -------------------------------------------

    def _chunk_position(self, match_id: str, orig: int) -> int | None:
        """Position in `chunk_frames` of the chunk with original source index `orig`, or None."""
        return self._chunk_pos.get(match_id, {}).get(orig)

    def _read_chunk(self, tar: tarfile.TarFile, entry: MatchEntry, ci: int) -> list[dict]:
        """Read one chunk's perspectives (`{"video", "lines"}`, optional `"physics"`) from an open tar.

        `ci` is the position into `chunk_frames`; the on-disk sample key uses the chunk's ORIGINAL
        source index (`entry.chunk_id(ci)`), which differs when the match has non-contiguous chunks."""
        key = chunk_key(entry.match_id, entry.chunk_id(ci))
        # Resolve members by basename (the streaming path does the same): WebDataset tars may store
        # members under a path prefix (`./key...` or a subdir), which `getmember(full_name)` misses.
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}

        def _member(name: str) -> tarfile.TarInfo:
            m = members.get(name)
            if m is None:
                raise ValueError(f"{entry.match_id}: tar member missing: {name}")
            return m

        out = []
        for i in range(entry.n_players):
            part = {
                "video": _read_member(tar, _member(f"{key}.p{i}.mp4")),
                "lines": _read_member(tar, _member(f"{key}.p{i}.jsonl")).splitlines(),
            }
            phys_name = f"{key}.p{i}.physics.jsonl"
            if phys_name in members:
                part["physics"] = _read_member(tar, members[phys_name]).splitlines()
            out.append(part)
        return out

    # -- public access ----------------------------------------------------

    def load_match(
        self,
        match_id: str,
        clip_len: int = 16,
        target_fps: int = 10,
        exclude_replays: bool = False,
        decode: bool = True,
        max_clips: int | None = None,
        perspective: str | int = "all",
        frame_size: tuple[int, int] | None = None,
        seed: int = 0,
        clip_ids: list[int] | None = None,
        action_fps: int | None = None,
    ) -> list[MatchClip]:
        """Random access: build the requested clips, reading only the chunks they need.

        A clip is taken from within a single chunk, so it is at most one chunk long (~80 frames at
        20 fps); a request where `clip_len` at the resulting stride exceeds the largest chunk raises.

        `action_fps` (additive; see `_assemble`) decouples the action sample rate from the frame
        `target_fps`; `None` keeps one action step per frame.
        """
        entry = self.matches[match_id]
        rng = random.Random(seed)
        mp = self._plan_match(entry, clip_len, target_fps)
        selected = self._select_plan(mp, exclude_replays, clip_ids, max_clips)

        needed = sorted({cp.chunk_idx for cp in selected})
        chunks: dict[int, list[dict]] = {}
        if needed:
            with tarfile.open(self.root / entry.shard, "r") as tar:
                for ci in needed:
                    chunks[ci] = self._read_chunk(tar, entry, ci)

        out = []
        for cp in selected:
            sel = self._select_perspectives(perspective, entry.n_players, rng)
            out.append(
                self._assemble(mp, cp, chunks[cp.chunk_idx], sel, target_fps, decode, frame_size, action_fps)
            )
        return out

    def iter_clips(
        self,
        clip_len: int = 16,
        target_fps: int = 10,
        exclude_replays: bool = False,
        decode: bool = True,
        perspective: str | int = "all",
        frame_size: tuple[int, int] | None = None,
        seed: int = 0,
        action_fps: int | None = None,
        carry_video: bool = False,
    ) -> Iterator[MatchClip]:
        """Stream shards, yielding one decoded clip at a time (each sample = one (match, chunk)).

        `action_fps` (additive; see `_assemble`) decouples the action sample rate from the frame
        `target_fps`; `None` keeps one action step per frame. `carry_video` attaches each clip's
        compressed per-perspective mp4 bytes (`MatchClip.video_bytes`) so decoding can be deferred;
        use it with `decode=False` to stream undecoded clips that a consumer decodes on demand.
        """
        rng = random.Random(seed)
        plans: dict[str, _MatchPlan] = {}  # cache per-match plan within a run
        failed: set[str] = set()  # matches whose plan raised; skipped (logged once) rather than fatal
        for shard in sorted({e.shard for e in self.index.entries}):
            yield from self._stream_shard(
                shard,
                plans,
                failed,
                clip_len,
                target_fps,
                exclude_replays,
                decode,
                perspective,
                frame_size,
                rng,
                action_fps,
                carry_video,
            )

    def _stream_shard(
        self,
        shard,
        plans,
        failed,
        clip_len,
        target_fps,
        exclude_replays,
        decode,
        perspective,
        frame_size,
        rng,
        action_fps=None,
        carry_video=False,
    ) -> Iterator[MatchClip]:
        def emit(key: str, parts: dict[int, dict]) -> Iterator[MatchClip]:
            parsed = parse_chunk_key(key)
            if parsed is None:
                return
            match_id, orig = parsed  # `orig` is the chunk's ORIGINAL source index (from the key)
            entry = self.matches.get(match_id)
            if entry is None or match_id in failed:
                return
            mp = plans.get(match_id)
            if mp is None:
                # A bad match (e.g. a clip too long for its chunks) should not abort the whole
                # stream; skip it and keep going, unlike load_match's explicit single-match request.
                try:
                    mp = plans[match_id] = self._plan_match(entry, clip_len, target_fps)
                except ValueError as err:
                    failed.add(match_id)
                    logger.warning("Skipping match %s while streaming: %s", match_id, err)
                    return
            pos = self._chunk_position(match_id, orig)  # map original index -> position in chunk_frames
            if pos is None:
                return
            # A single bad chunk (missing perspective member, short physics, decode failure) should
            # skip that chunk and keep the stream alive, the same way _plan_match errors skip a match
            # above -- otherwise one corrupt sample aborts the whole training run. Build the clips
            # eagerly so a failure is caught here rather than escaping the generator mid-iteration.
            try:
                chunk = [self._require_part(parts, i) for i in range(entry.n_players)]
                clips = []
                for cp in mp.plan:
                    if cp.chunk_idx != pos:
                        continue
                    if exclude_replays and overlaps_any(cp.g0, cp.g_end, mp.replays):
                        continue
                    sel = self._select_perspectives(perspective, entry.n_players, rng)
                    clips.append(
                        self._assemble(
                            mp, cp, chunk, sel, target_fps, decode, frame_size, action_fps, carry_video
                        )
                    )
            except (ValueError, IndexError, KeyError, RuntimeError, OSError) as err:
                logger.warning("Skipping chunk %s c%05d while streaming: %s", match_id, orig, err)
                return
            yield from clips

        cur_key: str | None = None
        parts: dict[int, dict] = {}
        with tarfile.open(self.root / shard, "r|*") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                key, field = _parse_member(Path(m.name).name)
                if key != cur_key:
                    if cur_key is not None:
                        yield from emit(cur_key, parts)
                    cur_key, parts = key, {}
                i = _perspective_index(field)
                if i is None:  # meta.json etc.
                    continue
                if field.endswith(".mp4"):
                    parts.setdefault(i, {})["video"] = _read_member(tar, m)
                elif field.endswith(".physics.jsonl"):  # check before the bare .jsonl below
                    parts.setdefault(i, {})["physics"] = _read_member(tar, m).splitlines()
                elif field.endswith(".jsonl"):
                    parts.setdefault(i, {})["lines"] = _read_member(tar, m).splitlines()
            if cur_key is not None:
                yield from emit(cur_key, parts)

    @staticmethod
    def _require_part(parts: dict[int, dict], i: int) -> dict:
        part = parts.get(i)
        if not part or "video" not in part or "lines" not in part:
            raise ValueError(f"chunk sample missing perspective {i} (have {sorted(parts)})")
        return part
