from __future__ import annotations

import math
import threading
import time
import unittest
from unittest import mock

from worldfoundry.studio.serving import jobs as jobs_module
from worldfoundry.studio.serving.jobs import StudioJob, StudioJobStore


class StudioJobStoreShutdownTest(unittest.TestCase):
    def test_studio_job_logs_use_central_secret_redaction(self) -> None:
        job = StudioJob(
            job_id="studio-00001",
            title="test",
            model_id="model",
            display_name="Model",
            action="run",
        )

        job.append_log(
            "stderr",
            "api_key=studio-secret Authorization: Bearer bearer-secret "
            "https://user:password@example.invalid\n",
        )

        stored = job.log_text()
        self.assertNotIn("studio-secret", stored)
        self.assertNotIn("bearer-secret", stored)
        self.assertNotIn("user:password", stored)
        self.assertIn("<redacted>", stored)

    def test_workspace_shutdown_returns_after_grace_and_cancels_queued_job(self) -> None:
        from worldfoundry.studio.serving import workspace as workspace_app

        started = threading.Event()
        release = threading.Event()
        store = StudioJobStore(max_workers=1)

        def blocking_run(_job):
            started.set()
            release.wait()
            return {"status": "completed"}

        running = store.submit_run(
            title="running",
            model_id="running",
            display_name="Running",
            action="run",
            metadata={},
            run_callable=blocking_run,
        )
        self.assertTrue(started.wait(timeout=2.0))
        queued = store.submit_run(
            title="queued",
            model_id="queued",
            display_name="Queued",
            action="run",
            metadata={},
            run_callable=lambda _job: {"status": "completed"},
        )

        try:
            with mock.patch.object(workspace_app, "JOBS", store), mock.patch.object(
                workspace_app, "_stop_all_visualizers"
            ) as stop_visualizers:
                before = time.monotonic()
                workspace_app._shutdown_workspace(grace_seconds=0.05)
                elapsed = time.monotonic() - before

            self.assertLess(elapsed, 1.0)
            self.assertTrue(running.cancel_requested)
            self.assertEqual(running.status, "running")
            self.assertTrue(queued.cancel_requested)
            self.assertEqual(queued.status, "cancelled")
            self.assertIsNotNone(queued._future)
            self.assertTrue(queued._future.cancelled())
            stop_visualizers.assert_called_once_with()
        finally:
            release.set()
            if running._future is not None:
                running._future.result(timeout=2.0)

        self.assertEqual(running.status, "cancelled")

    def test_shutdown_is_idempotent_and_rejects_new_jobs(self) -> None:
        store = StudioJobStore(max_workers=1)

        store.shutdown(grace_seconds=0)
        store.shutdown(grace_seconds=0)

        with self.assertRaisesRegex(RuntimeError, "cannot accept new jobs"):
            store.submit_run(
                title="late",
                model_id="late",
                display_name="Late",
                action="run",
                metadata={},
                run_callable=lambda _job: None,
            )

    def test_shutdown_grace_is_finite_non_negative_and_capped(self) -> None:
        self.assertLessEqual(
            jobs_module.DEFAULT_SHUTDOWN_GRACE_SECONDS,
            jobs_module.MAX_SHUTDOWN_GRACE_SECONDS,
        )
        self.assertLessEqual(jobs_module.MAX_SHUTDOWN_GRACE_SECONDS, 15.0)
        self.assertEqual(
            jobs_module._bounded_shutdown_grace_seconds(60.0),
            jobs_module.MAX_SHUTDOWN_GRACE_SECONDS,
        )
        for invalid in (-1.0, math.inf, -math.inf, math.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                jobs_module._bounded_shutdown_grace_seconds(invalid)

    def test_fastapi_uses_shared_workspace_shutdown_handler(self) -> None:
        from worldfoundry.studio.serving import workspace as workspace_app

        app = workspace_app.create_app()

        self.assertIn(workspace_app._shutdown_workspace, app.router.on_shutdown)


if __name__ == "__main__":
    unittest.main()
