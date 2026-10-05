from __future__ import annotations

from pathlib import Path


def _write_plugin_module(tmp_path: Path) -> None:
    (tmp_path / "fixture_pipeline_plugin.py").write_text(
        "\n".join(
            (
                "from worldfoundry.evaluation.models.pipelines.bindings import PipelineBinding",
                "",
                "BINDING = PipelineBinding(",
                "    binding_id='fixture-plugin',",
                "    model_id='fixture-plugin-model',",
                "    runner='worldfoundry.pipeline',",
                "    pipeline_target='json:loads',",
                "    aliases=('fixture-plugin-alias',),",
                ")",
                "",
                "def binding_factory():",
                "    return {",
                "        'schema_version': 2,",
                "        'binding_id': 'fixture-factory',",
                "        'model_id': 'fixture-factory-model',",
                "        'runner': 'worldfoundry.pipeline',",
                "        'pipeline': {'target': 'json:dumps'},",
                "    }",
                "",
            )
        ),
        encoding="utf-8",
    )


def test_pipeline_plugin_discovery_loads_env_binding_and_skips_broken_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from worldfoundry.evaluation.models.pipelines.discovery import ENV_VAR, discover_pipeline_bindings

    _write_plugin_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(
        ENV_VAR,
        "fixture=fixture_pipeline_plugin:BINDING,"
        "factory=fixture_pipeline_plugin:binding_factory,"
        "broken=missing_module:VALUE",
    )

    bindings = discover_pipeline_bindings()

    assert bindings["fixture"].model_id == "fixture-plugin-model"
    assert bindings["factory"].pipeline_target == "json:dumps"
    assert "broken" not in bindings


def test_pipeline_binding_resolution_includes_env_plugins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from worldfoundry.evaluation.models.pipelines.bindings import resolve_pipeline_binding
    from worldfoundry.evaluation.models.pipelines.discovery import ENV_VAR

    _write_plugin_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_VAR, "fixture=fixture_pipeline_plugin:BINDING")

    assert resolve_pipeline_binding("fixture-plugin", root=tmp_path / "empty").pipeline_target == "json:loads"
    assert resolve_pipeline_binding("fixture-plugin-alias", root=tmp_path / "empty").model_id == "fixture-plugin-model"


def test_pipeline_plugin_binding_does_not_shadow_builtin_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from worldfoundry.evaluation.models.pipelines.bindings import resolve_pipeline_binding
    from worldfoundry.evaluation.models.pipelines.discovery import ENV_VAR

    binding_root = tmp_path / "bindings" / "pipelines"
    binding_root.mkdir(parents=True)
    (binding_root / "fixture-plugin.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "binding_id: fixture-plugin",
                "model_id: fixture-builtin-model",
                "runner: worldfoundry.pipeline",
                "pipeline:",
                "  target: json:load",
                "aliases:",
                "  - fixture-builtin-alias",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_plugin_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_VAR, "fixture=fixture_pipeline_plugin:BINDING")

    binding = resolve_pipeline_binding("fixture-plugin", root=binding_root)

    assert binding.model_id == "fixture-builtin-model"
    assert binding.pipeline_target == "json:load"
