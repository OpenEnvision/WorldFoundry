"""Component pipeline exports remain lazy and resolve to existing public classes."""

from worldfoundry.pipelines import component_pipelines


def test_component_star_import_resolves_all_declared_exports() -> None:
    namespace = {}
    exec("from worldfoundry.pipelines.component_pipelines import *", namespace)
    for name in component_pipelines.__all__:
        assert namespace[name] is getattr(component_pipelines, name)
    assert not {"GaussianActorPipeline", "MultiTaskDiTPipeline", "TDMPCPipeline", "VQBeTPipeline"}.intersection(
        namespace
    )
