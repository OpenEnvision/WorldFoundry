"""Shared CUDA Graph and TeaCache mixins for stable-shape denoiser inner calls.

Most video denoisers apply family-specific rectified-flow / EDM preconditioning
around one call into an inner DiT (``self.model(...)``). That inner call has
stable input shapes across denoise steps and no data-dependent control flow, so
a CUDA Graph can capture the whole transformer stack and replay it launch-free.

This module is the opt-in seam between a recipe's ``RuntimePolicy.options`` /
``component_options`` and the runner's per-step ``DenoiserInput``.  It does not
change the ``Denoiser`` Protocol: the runner still calls
``denoiser(DenoiserInput) -> DenoiserOutput``.  A denoiser that opts in calls
:meth:`GraphWrappedDenoiserMixin._init_graph_runner` in its constructor and
routes the inner DiT through :meth:`_run_network`.  TeaCache kwargs come from
:meth:`FeatureCacheDenoiserMixin.feature_cache_kwargs`.

**CUDA Graph is opt-in and off by default.**
``resolve_cuda_graph_option`` reads ``component_options["cuda_graph"]`` first,
then ``policy.options["cuda_graph"]``, then ``False``.  When disabled,
``_graph_runner`` stays ``None`` and :meth:`_run_network` is a direct
``self.model(...)`` passthrough (zero overhead).  When enabled,
:class:`InferenceCUDAGraphRunner` still falls back to eager for CPU / grad /
unseen-signature inputs.

**TeaCache is opt-in and off by default.**
``resolve_teacache_threshold`` returns ``None`` unless ``teacache`` is a
positive float or ``True`` (default threshold ``0.15``).  ``teacache_thresh``
overrides the numeric threshold.  Training builds forbid the key
(``validate_runtime_policy_for_purpose``).  Threshold ``None`` yields empty
kwargs so the DiT stays on its dense, bitwise-unchanged path.

**Caches are isolated per request and CFG branch.** Classifier-free guidance
evaluates positive and negative residuals on alternating or paired calls, and
shared servers can interleave different prompts. Neither trajectory is
interchangeable. The mixin therefore keys cache ownership first by the
runner-created ``model_input.request_id`` and then by ``model_input.branch``
(``"positive"`` / ``"negative"``), never by a shared singleton. A
``data_ptr`` key would also be unsafe: allocators routinely reuse tensor
storage across branches and requests.

Denoisers whose inner call has variable-length sequences, host synchronizations
(``.item()``), Python list arguments, or data-dependent layout (e.g. the Cosmos3
omni transformer) are NOT graph-safe and must not use the graph mixin.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextvars import ContextVar
from threading import RLock
from typing import Any

from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

_CUDA_GRAPH_STATEFUL_CONFLICTS = (
    "adacache",
    "approximate_attention",
    "blocktaylorseer",
    "custom",
    "dualblock",
    "dynamicblock",
    "feature_cache",
    "firstblock",
    "magcache",
    "static_cross_kv",
    "taylorseer",
    "teacache",
    "teacache_thresh",
)


def validate_cuda_graph_options(context: Any) -> None:
    """Reject stateful options whose Python control flow cannot be captured.

    CUDA Graph replay executes the captured CUDA work only; it does not rerun
    Python cache invalidation, TeaCache skip decisions, or the per-step sparse
    attention schedule. Capturing any of those paths could freeze a residual or
    conditioning projection from a previous request. Rejecting the combination
    at build time is therefore a correctness requirement, not a performance
    preference.
    """

    if not resolve_cuda_graph_option(context):
        return
    conflicts: list[str] = []
    for name in _CUDA_GRAPH_STATEFUL_CONFLICTS:
        value = context.component_options.get(
            name,
            context.policy.options.get(name),
        )
        if value is not None and value is not False:
            conflicts.append(name)
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(
            "Wan cuda_graph cannot be combined with stateful cross-step options "
            f"({joined}); CUDA replay would bypass their request/step lifecycle"
        )


def resolve_cuda_graph_option(context: Any) -> bool:
    """Resolve the ``cuda_graph`` flag from a component build context.

    A component-level ``component_options["cuda_graph"]`` takes precedence over
    the run-wide ``policy.options["cuda_graph"]`` so a recipe can enable graphs
    for one component without turning them on globally.
    """

    return bool(context.component_options.get("cuda_graph", context.policy.options.get("cuda_graph", False)))


def resolve_teacache_threshold(context: Any) -> float | None:
    """Resolve the cross-step feature-cache threshold, or ``None`` (cache off).

    ``teacache`` accepts a float threshold directly, or ``True`` for a sane
    default; ``teacache_thresh`` overrides the threshold explicitly. Component
    options take precedence over the run-wide ``policy.options``. Shared by every
    DiT denoiser so the option surface is identical across backbones. Training
    builds never cache (``validate_runtime_policy_for_purpose`` forbids the key).
    """

    raw = context.component_options.get(
        "teacache", context.policy.options.get("teacache", None)
    )
    if raw is None or raw is False:
        return None
    thresh = context.component_options.get(
        "teacache_thresh", context.policy.options.get("teacache_thresh", None)
    )
    if thresh is not None:
        return float(thresh)
    if raw is True:
        return 0.15
    return float(raw)


class FeatureCacheDenoiserMixin:
    """Per-request, per-CFG-branch feature-cache ownership for a DiT.

    A denoiser sets ``self._teacache_threshold`` (from
    :func:`resolve_teacache_threshold`) in its constructor and calls
    :meth:`feature_cache_kwargs` to get the model kwargs to forward.

    Isolation is two-dimensional: an explicit runner-created ``request_id``
    selects a request state, then ``model_input.branch`` selects its cache.
    This survives A0/B0/A1 interleaving on a shared denoiser; step-zero or
    positive-first heuristics cannot provide that guarantee.  A ContextVar
    selects the report for the calling request without changing cache
    ownership.

    Threshold ``None`` returns an empty mapping, so the model runs its dense
    (bitwise-unchanged) path.  This mixin never mutates ``DenoiserInput``.
    """

    _teacache_threshold: float | None = None

    @property
    def _feature_cache_config(self) -> Any:
        config = self.__dict__.get("_configured_feature_cache")
        model = getattr(self, "model", None)
        plugin_config = getattr(model, "_worldfoundry_easycache_config", None)
        if plugin_config is not None:
            return plugin_config
        if getattr(config, "algorithm", None) == "easycache":
            return None
        return config

    @_feature_cache_config.setter
    def _feature_cache_config(self, value: Any) -> None:
        self.__dict__["_configured_feature_cache"] = value
        session = getattr(getattr(self, "model", None), "_worldfoundry_accelerations", None)
        if session is not None:
            session.bind_runtime(self)

    def _feature_cache_context_var(self) -> ContextVar[str | None]:
        context = self.__dict__.get("_feature_cache_current_request")
        if isinstance(context, ContextVar):
            return context
        context = ContextVar(
            f"worldfoundry_feature_cache_request_{id(self)}",
            default=None,
        )
        self.__dict__["_feature_cache_current_request"] = context
        return context

    def _feature_cache_lock(self) -> RLock:
        lock = self.__dict__.get("_feature_cache_state_lock")
        if lock is None:
            lock = RLock()
            self.__dict__["_feature_cache_state_lock"] = lock
        return lock

    @staticmethod
    def _explicit_request_id(model_input: Any) -> str:
        request_id = getattr(model_input, "request_id", None)
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError(
                "feature caching requires an explicit non-empty request_id; "
                "step/branch heuristics are not concurrency-safe"
            )
        return request_id

    def _feature_request_state(
        self,
        request_id: str,
        *,
        create: bool,
    ) -> dict[str, Any] | None:
        requests = self.__dict__.setdefault("_feature_cache_requests", {})
        with self._feature_cache_lock():
            state = requests.get(request_id)
            if state is None and create:
                epoch = int(self.__dict__.get("_feature_cache_epoch_counter", 0)) + 1
                self.__dict__["_feature_cache_epoch_counter"] = epoch
                state = {
                    "request_id": request_id,
                    "request_epoch": epoch,
                    "caches": {},
                    "branch_steps": {},
                    "complete": False,
                }
                requests[request_id] = state
                # Completed request receipts remain available for audit. Keep
                # a bounded history without ever evicting an active request.
                if len(requests) > 64:
                    completed = sorted(
                        (
                            candidate
                            for candidate in requests.values()
                            if bool(candidate.get("complete"))
                            and candidate.get("request_id") != request_id
                        ),
                        key=lambda candidate: int(candidate["request_epoch"]),
                    )
                    for candidate in completed[: max(len(requests) - 32, 0)]:
                        requests.pop(str(candidate["request_id"]), None)
            return state

    def feature_cache_kwargs(self, model_input: Any) -> dict[str, Any]:
        request_id = getattr(model_input, "request_id", None)
        if isinstance(request_id, str) and request_id.strip():
            self._feature_cache_context_var().set(request_id)
        session = getattr(getattr(self, "model", None), "_worldfoundry_accelerations", None)
        if session is not None:
            session.bind_runtime(self)
        config = getattr(self, "_feature_cache_config", None)
        threshold = getattr(self, "_teacache_threshold", None)
        if config is None and threshold is None:
            return {}
        if getattr(config, "algorithm", None) == "easycache":
            model = getattr(self, "model", None)
            if getattr(model, "training", False):
                raise RuntimeError("EasyCache requires model.eval()")
            if getattr(self, "_graph_runner", None) is not None or getattr(self, "inplace_residual", False):
                raise RuntimeError("EasyCache has not been validated with CUDA Graph or inplace residual")
            if threshold is not None:
                raise RuntimeError("EasyCache and legacy TeaCache are exclusive")
        request_id = self._explicit_request_id(model_input)
        state = self._feature_request_state(request_id, create=True)
        assert state is not None
        branch = str(model_input.branch)
        step = int(model_input.step_index)
        if step < 0:
            raise ValueError("feature-cache step must be non-negative")
        with self._feature_cache_lock():
            caches = state["caches"]
            cache = caches.get(branch)
            if cache is None:
                if step != 0:
                    raise ValueError(
                        "a request-local feature-cache branch must begin at step 0"
                    )
                if config is None:
                    # Backwards-compatible direct-constructor path used by
                    # older callers. Public builds use calibrated config.
                    from worldfoundry.core.acceleration import AdaptiveResidualCache

                    cache = AdaptiveResidualCache(
                        threshold,
                        total_steps=model_input.total_steps,
                    )
                elif config.algorithm == "easycache":
                    from worldfoundry.core.acceleration.easycache import EasyCache

                    cache = EasyCache(config, total_steps=model_input.total_steps)
                else:
                    from ...optimizations.wan.feature_cache import (
                        build_wan_feature_cache,
                    )

                    cache = build_wan_feature_cache(
                        config,
                        branch=branch,
                        total_steps=model_input.total_steps,
                    )
                cache.branch = branch
                cache.request_id = request_id
                cache.request_epoch = int(state["request_epoch"])
                caches[branch] = cache
            else:
                previous_step = state["branch_steps"].get(branch)
                if step == 0:
                    reset = getattr(cache, "reset", None)
                    if not callable(reset):
                        raise TypeError(
                            "request-local feature cache must implement reset()"
                        )
                    reset()
                elif previous_step is None or step != int(previous_step) + 1:
                    raise ValueError(
                        "feature-cache branch steps must be unique and contiguous"
                    )
            if getattr(cache, "algorithm", None) == "easycache":
                cache.observe_conditioning(model_input.conditioning)
            state["branch_steps"][branch] = step
            if step >= int(model_input.total_steps) - 1:
                state["complete"] = True
        return {
            "feature_cache": cache,
            "feature_cache_step": model_input.step_index,
            "feature_cache_total_steps": model_input.total_steps,
        }

    def feature_cache_report(self, request_id: str | None = None) -> dict[str, Any]:
        """Return telemetry for one explicit request, selected context-locally."""

        config = getattr(self, "_feature_cache_config", None)
        threshold = getattr(self, "_teacache_threshold", None)
        enabled = config is not None or threshold is not None
        algorithm = (
            str(config.algorithm)
            if config is not None
            else ("adaptive-residual-legacy" if enabled else "none")
        )
        branches: dict[str, dict[str, Any]] = {}
        total_hits = 0
        total_events = 0
        total_dense_block_calls = 0
        total_skipped_block_calls = 0
        if request_id is None:
            request_id = self._feature_cache_context_var().get()
        state = (
            self._feature_request_state(request_id, create=False)
            if isinstance(request_id, str)
            else None
        )
        if state is None and isinstance(request_id, str):
            encoded = self.__dict__.get("_feature_cache_receipt_snapshots", {}).get(
                request_id
            )
            if isinstance(encoded, str):
                decoded = json.loads(encoded)
                if not isinstance(decoded, dict):
                    raise TypeError("feature-cache receipt snapshot must decode to an object")
                return decoded
        request_epoch = int(state["request_epoch"]) if state is not None else 0
        caches = state["caches"] if state is not None else {}
        for branch, cache in caches.items():
            events = tuple(getattr(cache, "events", ()))
            hits = sum(bool(getattr(event, "hit", False)) for event in events)
            reasons: dict[str, int] = {}
            for event in events:
                reason = str(getattr(event, "reason", "unknown"))
                reasons[reason] = reasons.get(reason, 0) + 1
            total_hits += hits
            total_events += len(events)
            receipt_reporter = getattr(cache, "receipt", None)
            receipt = receipt_reporter() if callable(receipt_reporter) else None
            if not isinstance(receipt, Mapping):
                receipt = None
            dense_block_calls = getattr(cache, "dense_block_calls", 0)
            skipped_block_calls = getattr(cache, "skipped_block_calls", 0)
            if (
                isinstance(dense_block_calls, bool)
                or not isinstance(dense_block_calls, int)
                or dense_block_calls < 0
            ):
                dense_block_calls = 0
            if (
                isinstance(skipped_block_calls, bool)
                or not isinstance(skipped_block_calls, int)
                or skipped_block_calls < 0
            ):
                skipped_block_calls = 0
            total_dense_block_calls += dense_block_calls
            total_skipped_block_calls += skipped_block_calls
            cache_epoch = getattr(cache, "request_epoch", None)
            cache_branch = getattr(cache, "branch", None)
            cache_request_id = getattr(cache, "request_id", None)
            branch_report: dict[str, Any] = {
                "algorithm": str(getattr(cache, "algorithm", algorithm)),
                "request_id": cache_request_id,
                "request_epoch": cache_epoch,
                "request_local": cache_request_id == request_id
                and cache_epoch == request_epoch
                and str(cache_branch) == str(branch),
                "events": len(events),
                "hits": hits,
                "misses": len(events) - hits,
                "hit_rate": hits / len(events) if events else 0.0,
                "reasons": reasons,
                "dense_block_calls": dense_block_calls,
                "skipped_block_calls": skipped_block_calls,
            }
            if receipt is not None:
                branch_report["receipt"] = dict(receipt)
            branches[str(branch)] = branch_report
        if not enabled:
            effective = "disabled"
        elif total_events == 0:
            effective = "pending"
        elif total_hits:
            effective = "residual-reuse"
        else:
            effective = "dense-no-hit"
        return {
            "enabled": enabled,
            "algorithm": algorithm,
            "threshold": (
                config.options.get(
                    "residual_diff_threshold",
                    config.options.get("threshold"),
                )
                if config is not None
                else threshold
            ),
            "request_id": request_id,
            "request_epoch": request_epoch,
            "request_local": state is not None
            and bool(branches)
            and all(
                    bool(branch_report.get("request_local"))
                    for branch_report in branches.values()
                ),
            "effective": effective,
            "events": total_events,
            "hits": total_hits,
            "misses": total_events - total_hits,
            "hit_rate": total_hits / total_events if total_events else 0.0,
            "dense_block_calls": total_dense_block_calls,
            "skipped_block_calls": total_skipped_block_calls,
            "branches": branches,
        }

    def end_feature_cache_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Snapshot one receipt, then reset and release every tensor cache."""

        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("end_feature_cache_request requires a non-empty request_id")
        state = self._feature_request_state(request_id, create=False)
        if state is None:
            return
        report = self.feature_cache_report(request_id)
        report["finalized"] = True
        report["release_reason"] = "error" if error is not None else "completed"
        if error is not None:
            report["error_type"] = type(error).__name__

        with self._feature_cache_lock():
            requests = self.__dict__.setdefault("_feature_cache_requests", {})
            removed = requests.pop(request_id, None)
            if removed is None:
                return
            for cache in removed["caches"].values():
                reset = getattr(cache, "reset", None)
                if callable(reset):
                    reset()
            removed["caches"].clear()
            snapshots = self.__dict__.setdefault(
                "_feature_cache_receipt_snapshots",
                {},
            )
            snapshots[request_id] = json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
            )
            while len(snapshots) > 32:
                oldest_request_id = next(iter(snapshots))
                snapshots.pop(oldest_request_id, None)

    def feature_cache_lifecycle_report(self) -> dict[str, int]:
        """Expose bounded live/snapshot ownership without retaining tensors."""

        with self._feature_cache_lock():
            return {
                "live_requests": len(
                    self.__dict__.get("_feature_cache_requests", {})
                ),
                "receipt_snapshots": len(
                    self.__dict__.get("_feature_cache_receipt_snapshots", {})
                ),
                "max_receipt_snapshots": 32,
            }


class GraphWrappedDenoiserMixin:
    """Give a denoiser an opt-in CUDA Graph runner around its inner DiT call.

    ``enabled=False`` (the default from :func:`resolve_cuda_graph_option`)
    leaves ``_graph_runner`` unset.  :meth:`_run_network` then calls
    ``self.model`` with no extra frames.  ``graph_report`` returns ``None``
    when graphs are disabled so audit snapshots can distinguish "never
    opted in" from "opted in but still warming up".
    """

    _graph_runner: InferenceCUDAGraphRunner | None = None

    def _init_graph_runner(self, model: Any, *, enabled: bool, extra_key: str) -> None:
        self._graph_runner = (
            InferenceCUDAGraphRunner(model, extra_key=extra_key) if enabled else None
        )

    def _run_network(self, *args: Any, **kwargs: Any) -> Any:
        if self._graph_runner is not None:
            return self._graph_runner(*args, **kwargs)
        return self.model(*args, **kwargs)  # type: ignore[attr-defined]

    def _begin_graph_request_window(self) -> None:
        """Reset graph telemetry for a request without clearing graph cache."""

        if self._graph_runner is not None:
            self._graph_runner.begin_request_window()

    def graph_report(self) -> Mapping[str, Any] | None:
        """Return the runner's capture/replay stats, or ``None`` when disabled."""

        return self._graph_runner.report() if self._graph_runner is not None else None


__all__ = [
    "FeatureCacheDenoiserMixin",
    "GraphWrappedDenoiserMixin",
    "resolve_cuda_graph_option",
    "resolve_teacache_threshold",
    "validate_cuda_graph_options",
]
