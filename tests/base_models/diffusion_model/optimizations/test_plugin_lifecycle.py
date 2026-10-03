"""Plugin mutation respects dense requests, live forwards and graph ownership."""

from copy import deepcopy
from threading import Event, Thread
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
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WanDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
from worldfoundry.base_models.diffusion_model.runners.base import NativeDiffusionRunner, RunnerComponents
from worldfoundry.core.acceleration.plugins import AccelerationHandle, AccelerationRegistry, PreparedAcceleration


def _model():
    return WanModel(
        dim=12,
        in_dim=2,
        ffn_dim=24,
        out_dim=2,
        text_dim=8,
        freq_dim=4,
        eps=1e-6,
        patch_size=(1, 1, 1),
        num_heads=1,
        num_layers=1,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()


def _denoiser():
    return WanDenoiser(_model(), compute_dtype=torch.float32)


def _input(request="request-a", branch="positive"):
    return DenoiserInput(
        latents=torch.ones(1, 2, 1, 2, 2),
        timestep=torch.tensor([500.0]),
        next_timestep=torch.tensor([400.0]),
        conditioning={"context": torch.ones(1, 2, 8)},
        step_index=0,
        total_steps=1,
        request_id=request,
        branch=branch,
    )


def _install(model):
    return install_diffusion_accelerations(model, {"attention_policy": {"self": "torch"}})


def test_install_cannot_change_a_dense_request_started_before_any_plugin():
    denoiser = _denoiser()
    keys = tuple(denoiser.model.state_dict())
    with torch.inference_mode():
        denoiser(_input())
    assert denoiser.feature_cache_lifecycle_report()["live_requests"] == 0
    with pytest.raises(RuntimeError, match="active requests"):
        _install(denoiser.model)
    denoiser.end_request("request-a")
    session = _install(denoiser.model)
    assert tuple(denoiser.model.state_dict()) == keys
    session.uninstall()


def test_attention_only_removal_waits_for_every_request_and_cfg_branch():
    denoiser = _denoiser()
    session = _install(denoiser.model)
    with torch.inference_mode():
        for request, branch in (("a", "negative"), ("b", "positive"), ("a", "positive")):
            denoiser(_input(request, branch))
    with pytest.raises(RuntimeError, match="active requests"):
        session.uninstall()
    denoiser.end_request("a")
    with pytest.raises(RuntimeError, match="active requests"):
        session.uninstall()
    denoiser.end_request("b")
    session.uninstall()


def test_direct_forward_without_request_id_is_guarded_until_it_returns():
    denoiser = _denoiser()
    entered, release = Event(), Event()
    errors = []

    def pause(module, args, kwargs):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test forward timed out")

    hook = denoiser.model.register_forward_pre_hook(pause, with_kwargs=True)

    def run():
        try:
            with torch.inference_mode():
                denoiser(_input(None))
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="active requests"):
            _install(denoiser.model)
    finally:
        release.set()
        worker.join(5)
        hook.remove()
    assert not worker.is_alive() and not errors
    _install(denoiser.model).uninstall()


def test_concurrent_cfg_forwards_still_guard_removal_after_request_finalization():
    denoiser = _denoiser()
    session = _install(denoiser.model)
    entered = {branch: Event() for branch in ("positive", "negative")}
    release, errors = Event(), []

    def pause(module, args, kwargs):
        branch = "positive" if kwargs["context"].shape[1] == 2 else "negative"
        entered[branch].set()
        if not release.wait(5):
            raise RuntimeError("test forward timed out")

    # Distinguish the two calls without depending on thread ordering.
    hook = denoiser.model.register_forward_pre_hook(pause, with_kwargs=True)

    def run(branch):
        try:
            item = _input(branch=branch)
            if branch == "negative":
                item = item.with_updates(conditioning={"context": torch.ones(1, 3, 8)})
            with torch.inference_mode():
                denoiser(item)
        except BaseException as error:
            errors.append(error)

    workers = [Thread(target=run, args=(branch,)) for branch in entered]
    for worker in workers:
        worker.start()
    try:
        assert all(event.wait(5) for event in entered.values())
        denoiser.end_request("request-a")
        with pytest.raises(RuntimeError, match="active requests"):
            session.uninstall()
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
        hook.remove()
    assert not any(worker.is_alive() for worker in workers) and not errors
    session.uninstall()


@pytest.mark.parametrize("install_before_graph", [False, True])
def test_graph_ownership_blocks_mutation_even_before_first_capture(install_before_graph):
    denoiser = _denoiser()
    session = _install(denoiser.model) if install_before_graph else None
    denoiser._init_graph_runner(denoiser.model, enabled=True, extra_key="test")
    with pytest.raises(RuntimeError, match="CUDA Graph"):
        session.uninstall() if session is not None else _install(denoiser.model)


@pytest.mark.parametrize("failure_stage", ["denoiser", "scheduler"])
def test_native_runner_releases_dense_request_on_failure(failure_stage):
    denoiser = _denoiser()
    session = _install(denoiser.model)

    def fail(*args, **kwargs):
        raise RuntimeError(f"{failure_stage} failed")

    hook = denoiser.model.register_forward_pre_hook(fail) if failure_stage == "denoiser" else None
    scheduler = SimpleNamespace(
        schedule=lambda sampling, **kwargs: (SchedulerStep(0, torch.tensor(500.0), torch.tensor(0.0)),),
        scale_model_input=lambda latents, step: latents,
        step=fail,
    )
    runner = NativeDiffusionRunner(
        model_id="native-wan-lifecycle-test",
        guidance_mode="positive",
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=SimpleNamespace(encode=lambda *args, **kwargs: Conditioning(positive=_input().conditioning)),
            latent_initializer=SimpleNamespace(initialize=lambda *args, **kwargs: _input().latents),
            scheduler=scheduler,
            decoder=SimpleNamespace(decode=lambda latents, request: latents),
        ),
    )
    try:
        with torch.inference_mode(), pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            runner.run(DiffusionRequest(prompt="test", sampling=SamplingConfig(num_inference_steps=1)))
    finally:
        if hook is not None:
            hook.remove()
    session.uninstall()


def test_model_copy_has_independent_runtime_ownership_and_no_tensor_keys():
    denoiser = _denoiser()
    with torch.inference_mode():
        denoiser(_input())
    copied = deepcopy(denoiser.model)
    assert tuple(copied.state_dict()) == tuple(denoiser.model.state_dict())
    _install(copied).uninstall()
    with pytest.raises(RuntimeError, match="active requests"):
        _install(denoiser.model)
    denoiser.end_request("request-a")


def test_easycache_rejects_duplicate_step_zero_without_resetting_request_history():
    denoiser = _denoiser()
    session = install_diffusion_accelerations(denoiser.model, {"easycache": {"threshold": 0.1}})
    with torch.inference_mode():
        denoiser(_input().with_updates(total_steps=3))
        with pytest.raises(ValueError, match="unique and contiguous"):
            denoiser(_input().with_updates(total_steps=3))
    assert denoiser.feature_cache_report("request-a")["events"] == 1
    denoiser.end_request("request-a")
    session.uninstall()


def test_child_plugins_follow_dense_parent_request_ownership():
    denoiser = _denoiser()
    child = denoiser.model.blocks[0].cross_attn
    session = install_diffusion_accelerations(child, {"wan_cross_kv_fusion": {"min_tokens": 0}})
    with torch.inference_mode():
        denoiser(_input())
    with pytest.raises(RuntimeError, match="active requests"):
        session.uninstall()
    denoiser.end_request("request-a")
    session.uninstall()
    with torch.inference_mode():
        denoiser(_input("b"))
    with pytest.raises(RuntimeError, match="active requests"):
        install_diffusion_accelerations(child, {"wan_cross_kv_fusion": True})
    denoiser.end_request("b")
    install_diffusion_accelerations(child, {"wan_cross_kv_fusion": True}).uninstall()


def test_idle_isolated_child_session_can_bind_to_a_later_native_parent():
    model = _model()
    child = model.blocks[0].cross_attn
    session = install_diffusion_accelerations(child, {"wan_cross_kv_fusion": True})
    denoiser = WanDenoiser(model, compute_dtype=torch.float32)
    with torch.inference_mode():
        denoiser(_input())
    with pytest.raises(RuntimeError, match="active requests"):
        session.uninstall()
    denoiser.end_request("request-a")
    session.uninstall()


def test_registered_child_owner_is_not_silently_migrated_to_another_root():
    model = _model()
    child_owner = WanDenoiser(model.blocks[0].cross_attn, compute_dtype=torch.float32)
    with pytest.raises(RuntimeError, match="another registered"):
        WanDenoiser(model, compute_dtype=torch.float32)
    assert child_owner.model is model.blocks[0].cross_attn


@pytest.mark.parametrize("uninstall", [False, True])
def test_parent_compile_marker_blocks_child_plugin_mutation(uninstall):
    denoiser = _denoiser()
    child = denoiser.model.blocks[0].cross_attn
    session = install_diffusion_accelerations(child, {"wan_cross_kv_fusion": True}) if uninstall else None
    denoiser.model._worldfoundry_compile_runtime = {}
    with pytest.raises(RuntimeError, match="compiled model"):
        session.uninstall() if session else install_diffusion_accelerations(child, {"wan_cross_kv_fusion": True})


def test_parent_registration_fails_closed_without_waiting_on_child_installer():
    model = _model()
    child = model.blocks[0].cross_attn
    entered, release, registration_done = Event(), Event(), Event()
    install_errors, registration_errors, sessions = [], [], []
    registry = AccelerationRegistry()

    def prepare(target, options, policy):
        def activate():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test installer timed out")
            return AccelerationHandle("test", {}, lambda: None)

        return PreparedAcceleration("test", frozenset(), activate)

    registry.register("test", prepare)

    def install():
        try:
            sessions.append(registry.install(child, {"test": True}))
        except BaseException as error:
            install_errors.append(error)

    def register_parent():
        try:
            WanDenoiser(model, compute_dtype=torch.float32)
        except BaseException as error:
            registration_errors.append(error)
        finally:
            registration_done.set()

    installer = Thread(target=install)
    registrar = Thread(target=register_parent)
    installer.start()
    try:
        assert entered.wait(5)
        registrar.start()
        assert registration_done.wait(2)
        assert len(registration_errors) == 1
        assert "mutating" in str(registration_errors[0])
    finally:
        release.set()
        installer.join(5)
        if registrar.ident is not None:
            registrar.join(5)
    assert not installer.is_alive() and not registrar.is_alive() and not install_errors
    # Once the foreign transaction finishes, an idle ownership rebind works.
    denoiser = WanDenoiser(model, compute_dtype=torch.float32)
    sessions[0].uninstall()
    assert denoiser.model is model


def test_new_plugin_replacements_inherit_the_registered_root_guard():
    denoiser = _denoiser()
    model = denoiser.model
    registry = AccelerationRegistry()
    original = model.blocks[0].ffn[0]

    def prepare(target, options, policy):
        def activate():
            target.blocks[0].ffn[0] = torch.nn.Identity()
            return AccelerationHandle("replace", {}, lambda: target.blocks[0].ffn.__setitem__(0, original))

        return PreparedAcceleration("replace", frozenset(), activate)

    registry.register("replace", prepare)
    session = registry.install(model, {"replace": True})
    replacement = model.blocks[0].ffn[0]
    assert (
        replacement._worldfoundry_acceleration_runtime_ownership is model._worldfoundry_acceleration_runtime_ownership
    )
    # No model forward is needed to establish the new descendant's guard.
    from worldfoundry.core.acceleration.plugins import acceleration_runtime_scope

    with acceleration_runtime_scope(model, denoiser, "a"):
        with pytest.raises(RuntimeError, match="active requests"):
            AccelerationRegistry().install(replacement, {})
    denoiser.end_request("a")
    session.uninstall()
