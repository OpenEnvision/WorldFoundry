"""Real tiny Wan experts qualify local cache boundaries and global routing."""

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionRequest,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import Wan22DualExpertDenoiser, WanDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
from worldfoundry.base_models.diffusion_model.runners.base import (
    RunnerComponents,
    Wan22DualExpertGuidanceRunner,
)
from worldfoundry.core.model_loading.policy import RuntimePolicy


def _experts():
    experts = []
    for seed in (9, 19):
        torch.manual_seed(seed)
        model = WanModel(
            dim=12,
            in_dim=2,
            ffn_dim=24,
            out_dim=2,
            text_dim=8,
            freq_dim=4,
            eps=1e-6,
            patch_size=(1, 1, 1),
            num_heads=1,
            num_layers=2,
            has_image_input=False,
            require_vae_embedding=False,
            require_clip_embedding=False,
        ).eval()
        experts.append(WanDenoiser(model, compute_dtype=torch.float32))
    return experts


def _schedule(times=(950, 925, 900, 875, 800, 600, 400, 200)):
    return tuple(
        SchedulerStep(index, torch.tensor(float(time)), torch.tensor(float(next_time)))
        for index, (time, next_time) in enumerate(zip(times, (*times[1:], 0), strict=True))
    )


def _input(schedule, index=0, branch="positive", request="a"):
    step = schedule[index]
    return DenoiserInput(
        latents=torch.ones(1, 2, 1, 2, 2),
        timestep=step.timestep,
        next_timestep=step.next_timestep,
        conditioning={"context": torch.full((1, 2, 8), 1.0 if branch == "positive" else 2.0)},
        step_index=index,
        total_steps=len(schedule),
        request_id=request,
        branch=branch,
    )


def _install(experts, threshold):
    return [
        install_diffusion_accelerations(expert.model, {"easycache": {"threshold": threshold}}) for expert in experts
    ]


def _runner(denoiser, schedule, *, fail_scheduler=False, high_scale=4.0, low_scale=3.0):
    def scheduler_step(model_output, step, latents, **kwargs):
        if fail_scheduler:
            raise RuntimeError("scheduler failed")
        return latents

    return Wan22DualExpertGuidanceRunner(
        model_id="native-dual-wan-easycache-test",
        boundary_ratio=0.875,
        high_noise_guidance_scale=high_scale,
        low_noise_guidance_scale=low_scale,
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=SimpleNamespace(
                encode=lambda *args, **kwargs: Conditioning(
                    positive=_input(schedule).conditioning,
                    negative=_input(schedule, branch="negative").conditioning,
                )
            ),
            latent_initializer=SimpleNamespace(initialize=lambda *args, **kwargs: _input(schedule).latents),
            scheduler=SimpleNamespace(
                schedule=lambda sampling, **kwargs: schedule,
                scale_model_input=lambda latents, step: latents,
                step=scheduler_step,
            ),
            decoder=SimpleNamespace(decode=lambda latents, request: latents),
        ),
    )


@pytest.mark.parametrize("threshold", [0.0, 10.0])
def test_dual_expert_caches_have_local_warmup_terminal_dense_and_separate_cfg(threshold):
    experts = _experts()
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule()
    expected = {}
    with torch.inference_mode():
        for index in range(len(schedule)):
            for branch in ("negative", "positive"):
                expected[index, branch] = denoiser(_input(schedule, index, branch, "dense")).sample
    denoiser.end_request("dense")
    sessions = _install(experts, threshold)
    denoiser.prepare_request_schedule("a", schedule)
    with torch.inference_mode():
        for index in range(len(schedule)):
            for branch in ("negative", "positive"):
                actual = denoiser(_input(schedule, index, branch)).sample
                if threshold == 0:
                    torch.testing.assert_close(actual, expected[index, branch], rtol=0, atol=0)
    for expert in experts:
        report = expert.feature_cache_report("a")
        assert set(report["branches"]) == {"positive", "negative"}
        caches = expert._feature_cache_requests["a"]["caches"]
        assert caches["positive"] is not caches["negative"]
        for cache in caches.values():
            assert cache.total_steps == 4
            assert [event.step for event in cache.events] == [0, 1, 2, 3]
            assert [event.reason for event in cache.events[:2]] == ["warmup", "warmup"]
            assert cache.events[-1].reason == "dense-last"
            assert sum(event.hit for event in cache.events) == (0 if threshold == 0 else 1)
        assert report["dense_block_calls"] + report["skipped_block_calls"] == 16
    receipt = denoiser.route_receipt("a")
    assert [event["step_index"] for event in receipt["events"]] == [i for i in range(8) for _ in range(2)]
    assert [event["local_step_index"] for event in receipt["events"]] == [
        i for _ in range(2) for i in range(4) for _ in range(2)
    ]
    assert receipt["expert_schedule"]["phases"][1]["global_start"] == 4
    denoiser.end_request("a")
    assert denoiser.route_lifecycle_report()["live_requests"] == 0
    for expert, session in zip(experts, sessions, strict=True):
        assert expert.feature_cache_lifecycle_report()["live_requests"] == 0
        session.uninstall()


@pytest.mark.parametrize("times", [(950,), (800,), (950, 900), (800, 400), (950, 800)])
def test_short_or_single_phase_schedule_keeps_every_step_dense(times):
    experts = _experts()
    sessions = _install(experts, 10.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule(times)
    denoiser.prepare_request_schedule("a", schedule)
    with torch.inference_mode():
        for index in range(len(schedule)):
            denoiser(_input(schedule, index))
    denoiser.end_request("a")
    for expert, session in zip(experts, sessions, strict=True):
        assert expert.feature_cache_report("a")["hits"] == 0
        session.uninstall()


def test_negative_cfg_branch_can_start_only_at_low_expert_boundary():
    experts = _experts()
    sessions = _install(experts, 10.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule()
    denoiser.prepare_request_schedule("a", schedule)
    with torch.inference_mode():
        for index in range(len(schedule)):
            denoiser(_input(schedule, index))
            if index >= 4:
                denoiser(_input(schedule, index, "negative"))
    assert set(experts[0].feature_cache_report("a")["branches"]) == {"positive"}
    assert set(experts[1].feature_cache_report("a")["branches"]) == {"positive", "negative"}
    denoiser.end_request("a")
    for session in sessions:
        session.uninstall()


def test_interleaved_requests_never_share_expert_branch_caches():
    experts = _experts()
    sessions = _install(experts, 10.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule()
    for request in ("a", "b"):
        denoiser.prepare_request_schedule(request, schedule)
    with torch.inference_mode():
        for index in range(len(schedule)):
            for request in ("a", "b"):
                item = _input(schedule, index, request=request)
                if request == "b":
                    item = item.with_updates(latents=item.latents + 2)
                denoiser(item)
    for expert in experts:
        a = expert._feature_cache_requests["a"]["caches"]["positive"]
        b = expert._feature_cache_requests["b"]["caches"]["positive"]
        assert a is not b and a.request_epoch != b.request_epoch
        assert not torch.equal(a._dense_input, b._dense_input)
    denoiser.end_request("a")
    with pytest.raises(RuntimeError, match="active requests"):
        sessions[0].uninstall()
    denoiser.end_request("b")
    for session in sessions:
        session.uninstall()


@pytest.mark.parametrize("bad_index", [0, 2])
def test_duplicate_or_gapped_branch_calls_fail_without_advancing_receipt(bad_index):
    experts = _experts()
    sessions = _install(experts, 10.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule()
    denoiser.prepare_request_schedule("a", schedule)
    with torch.inference_mode():
        denoiser(_input(schedule))
        with pytest.raises(ValueError, match="unique and contiguous") as caught:
            denoiser(_input(schedule, bad_index))
    assert len(denoiser.route_receipt("a")["events"]) == 1
    denoiser.end_request("a", error=caught.value)
    assert denoiser.route_receipt("a")["release_reason"] == "error"
    for session in sessions:
        session.uninstall()


def test_missing_changed_and_incomplete_schedule_are_rejected_and_released():
    experts = _experts()
    sessions = _install(experts, 10.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule()
    with pytest.raises(ValueError, match="prepared full"):
        denoiser(_input(schedule))
    denoiser.prepare_request_schedule("a", schedule)
    with pytest.raises(RuntimeError, match="active requests"):
        sessions[1].uninstall()
    with pytest.raises(ValueError, match="schedule changed"):
        denoiser.prepare_request_schedule("a", _schedule((950, 800)))
    with pytest.raises(ValueError, match="timesteps changed"):
        denoiser(_input(schedule).with_updates(timestep=torch.tensor(940.0)))
    with torch.inference_mode():
        denoiser(_input(schedule))
    with pytest.raises(ValueError, match="phases completed"):
        denoiser.end_request("a")
    assert denoiser.route_lifecycle_report()["live_requests"] == 0
    for session in sessions:
        session.uninstall()


@pytest.mark.parametrize("times", [(800, 950), (950, 800, 925), (float("nan"),), (float("inf"),)])
def test_malformed_schedule_rejected_before_any_request_state(times):
    experts = _experts()
    sessions = _install(experts, 10.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    with pytest.raises(ValueError, match="schedule"):
        denoiser.prepare_request_schedule("a", _schedule(times))
    assert denoiser.route_lifecycle_report()["live_requests"] == 0
    for session in sessions:
        session.uninstall()


def test_shared_expert_models_and_sparse_cfg_execution_are_rejected():
    experts = _experts()
    sessions = _install(experts, 0.0)
    with pytest.raises(ValueError, match="independent"):
        Wan22DualExpertDenoiser(experts[0], experts[0], boundary_ratio=0.875)
    shared = WanDenoiser(experts[0].model, compute_dtype=torch.float32)
    with pytest.raises(ValueError, match="independent"):
        Wan22DualExpertDenoiser(experts[0], shared, boundary_ratio=0.875)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    for degree, gate in ((2, 1.0), (1, 0.5)):
        with pytest.raises(ValueError, match="dense single-rank CFG"):
            denoiser.validate_request_execution(cfg_parallel_degree=degree, cfg_gate_step=gate)
    for session in sessions:
        session.uninstall()


def test_native_runner_prepares_plan_and_releases_both_experts_on_failure():
    experts = _experts()
    sessions = _install(experts, 0.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule((950, 800))

    runner = _runner(denoiser, schedule, fail_scheduler=True)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="scheduler failed"):
        runner.run(DiffusionRequest(prompt="test", sampling=SamplingConfig(num_inference_steps=2)))
    assert denoiser.route_lifecycle_report()["live_requests"] == 0
    assert denoiser.route_lifecycle_report()["receipt_snapshots"] == 1
    for expert, session in zip(experts, sessions, strict=True):
        assert expert.feature_cache_lifecycle_report()["live_requests"] == 0
        session.uninstall()


@pytest.mark.parametrize("high_scale,low_scale", [(4.0, 3.0), (1.0, 3.0), (4.0, 1.0)])
def test_released_wan_guidance_runner_uses_full_plan_and_skips_unused_cfg_phases(high_scale, low_scale):
    experts = _experts()
    sessions = _install(experts, 0.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule()
    runner = _runner(denoiser, schedule, high_scale=high_scale, low_scale=low_scale)
    with torch.inference_mode():
        runner.run(DiffusionRequest(prompt="test", sampling=SamplingConfig(num_inference_steps=8)))
    receipt = denoiser.route_receipt()
    assert receipt["finalized"] and receipt["release_reason"] == "completed"
    assert receipt["route_calls"] == {
        "high-noise": 4 if high_scale == 1 else 8,
        "low-noise": 4 if low_scale == 1 else 8,
    }
    for expert, scale, session in zip(experts, (high_scale, low_scale), sessions, strict=True):
        assert set(expert.feature_cache_report(receipt["request_id"])["branches"]) == (
            {"positive"} if scale == 1 else {"positive", "negative"}
        )
        session.uninstall()


def test_schedule_snapshots_detect_tensor_mutation_and_broken_intervals():
    experts = _experts()
    sessions = _install(experts, 0.0)
    denoiser = Wan22DualExpertDenoiser(*experts, boundary_ratio=0.875)
    schedule = _schedule()
    denoiser.prepare_request_schedule("a", schedule)
    schedule[0].timestep.fill_(940.0)
    with pytest.raises(ValueError, match="timesteps changed") as caught:
        denoiser(_input(schedule))
    denoiser.end_request("a", error=caught.value)
    malformed = list(_schedule())
    malformed[1] = SchedulerStep(2, malformed[1].timestep, malformed[1].next_timestep)
    with pytest.raises(ValueError, match="indices"):
        denoiser.prepare_request_schedule("bad-index", malformed)
    malformed = list(_schedule())
    malformed[0] = SchedulerStep(0, malformed[0].timestep, torch.tensor(850.0))
    with pytest.raises(ValueError, match="matching next"):
        denoiser.prepare_request_schedule("bad-interval", malformed)
    for session in sessions:
        session.uninstall()


def test_native_factory_forwards_plugins_to_both_independent_checkpoint_roles(monkeypatch):
    from worldfoundry.base_models.diffusion_model.models.denoisers import wan

    experts = iter(_experts())
    roles, sessions = [], []

    def build(context, **kwargs):
        roles.append((context.key.name, context.require_checkpoint("weights").source[0]))
        expert = next(experts)
        sessions.append(install_diffusion_accelerations(expert.model, context.policy.options["accelerations"]))
        return expert

    monkeypatch.setattr(wan, "_build_wan_denoiser", build)
    context = SimpleNamespace(
        model_id="test-wan22",
        policy=RuntimePolicy(options={"accelerations": {"easycache": {"threshold": 0.0}}}),
        component_options={},
        purpose="inference",
        recipe_options={},
        require_checkpoint=lambda role: CheckpointSpec(source=role),
    )
    denoiser = wan._build_wan22_dual_expert_denoiser(context, config={})
    assert roles == [("high-noise", "high_weights"), ("low-noise", "low_weights")]
    assert denoiser.high_noise.model is not denoiser.low_noise.model
    for session in sessions:
        session.uninstall()
