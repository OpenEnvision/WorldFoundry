"""The documented programmatic entry points resolve from worldfoundry.evals.public."""


def test_public_entry_points_resolve():
    from worldfoundry.evals import public

    assert callable(public.run_worldfoundry)
    assert callable(public.run_model_benchmark)
    assert callable(public.execute_evaluate_run)
    assert callable(public.run_benchmark)
