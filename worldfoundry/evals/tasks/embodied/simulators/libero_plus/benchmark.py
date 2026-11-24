"""LIBERO-Plus benchmark implementation.

LIBERO-Plus (https://github.com/sylvestf/LIBERO-plus) is a robustness-analysis
extension of LIBERO that replaces each of the original 40 evaluation tasks
with ~10,030 systematically perturbed variants across seven axes: object
layout, camera viewpoints, robot initial states, language instructions,
lighting, background textures, and sensor noise.

Because the fork installs under the same ``libero`` package namespace as
vanilla LIBERO and registers suites under identical names
(``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``,
``libero_90``), this class is a thin subclass of :class:`LIBEROBenchmark`
that delegates all env/observation/action logic to the parent and only adds
task filtering using ``benchmark/task_classification.json``.

The ``libero`` and ``libero-plus`` packages cannot coexist in the same
Python environment; run this benchmark from the dedicated libero-plus Docker
image (see the runtime profile for the registry URL).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.embodied.simulators.libero.benchmark import LIBEROBenchmark


def _registry_name(task: dict[str, Any]) -> str | None:
    """Return LIBERO's internal registry id (``task_obj.name``) for *task*.

    LIBEROBenchmark stores the human-readable ``task.language`` under
    ``task["name"]``; ``task_classification.json`` is keyed by the
    registry id, so filters and metadata joins must use this.

    Args:
        task: A dictionary containing the task metadata and task_obj.

    Returns:
        The registry ID string if found, otherwise None.
    """
    return getattr(task.get("task_obj"), "name", None)


class LIBEROPlusBenchmark(LIBEROBenchmark):
    """LIBERO-Plus robustness benchmark.

    Accepts every keyword argument :class:`LIBEROBenchmark` accepts (forwarded
    via ``**kwargs``) plus LIBERO-Plus-specific filters:

    Args:
        category: Optional filter on ``task_classification.json`` category
            (e.g. ``"Background Textures"``, ``"Camera Viewpoints"``). When
            set, only task variants tagged with this category are returned.
            ``libero_90`` has no classification metadata and accepts only
            ``category=None``.
        difficulty_level: Optional filter on ``difficulty_level`` (integer,
            typically 1-3). Combined with *category* via logical AND.

    Use the orchestrator's top-level ``max_tasks:`` config key to limit the
    task count after filtering.
    """

    def __init__(
        self,
        *,
        category: str | None = None,
        difficulty_level: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LIBERO-Plus benchmark with optional filtering criteria.

        Args:
            category: Optional filter for the task classification category (e.g., "Background Textures").
            difficulty_level: Optional filter for the task difficulty level (integer, typically 1-3).
            **kwargs: Arbitrary keyword arguments passed to the parent LIBEROBenchmark
                      constructor, such as `suite`.
        """
        super().__init__(**kwargs)
        self.category = category
        self.difficulty_level = difficulty_level
        self._classification: dict[str, dict[str, Any]] | None = None

    def _load_classification(self) -> dict[str, dict[str, Any]]:
        """Load and index task_classification.json as ``{task_name: entry}``.

        This method lazily loads the classification data from the `libero` package
        and caches it.

        Returns:
            A dictionary mapping task names (registry IDs) to their classification
            entry records, or an empty dictionary if the file is not found.
        """
        if self._classification is not None:
            return self._classification

        # Lazy import `libero` package because it is only available inside the
        # benchmark Docker image, ensuring the benchmark runs outside without it.
        libero_benchmark = __import__("libero.libero", fromlist=["benchmark"]).benchmark

        # Determine the path to the task_classification.json file relative to the libero benchmark package.
        benchmark_dir = Path(libero_benchmark.__file__).parent
        classification_path = benchmark_dir / "task_classification.json"

        try:
            # Attempt to open and load the JSON classification file.
            with open(classification_path) as f:
                raw = json.load(f)
        except FileNotFoundError:
            # If the file is not found, initialize classification as an empty dictionary.
            self._classification = {}
            return self._classification

        # Index the raw classification data by task name (registry ID) for efficient lookup.
        # It filters entries specific to the current benchmark suite and ensures a 'name' key exists.
        self._classification = {entry["name"]: entry for entry in raw.get(self.suite, []) if entry.get("name")}
        return self._classification

    def get_tasks(self) -> list[dict[str, Any]]:
        """Retrieve and filter the LIBERO-Plus robustness evaluation tasks.

        This method first retrieves tasks from the parent LIBEROBenchmark and then
        applies category and difficulty level filters according to classification
        metadata defined in `task_classification.json`. It also enriches task
        dictionaries with classification metadata if available.

        Returns:
            A list of filtered task dictionaries, potentially enriched with robustness metadata.

        Raises:
            RuntimeError: If filters are set but no classification metadata is found
                          for the configured benchmark suite.
        """
        tasks = super().get_tasks()
        classification = self._load_classification()

        # Determine if any classification-based filtering is required by checking the presence of filters.
        needs_classification = self.category is not None or self.difficulty_level is not None
        if needs_classification and not classification:
            # Raise an error if filters are set but classification data is missing for the suite,
            # as this indicates misconfiguration or an unsupported suite.
            raise RuntimeError(
                f"category/difficulty_level filter set but no classification metadata found "
                f"for suite {self.suite!r} (task_classification.json covers "
                f"libero_spatial/object/goal/10 only)."
            )

        filtered: list[dict[str, Any]] = []
        for task in tasks:
            # Get the classification entry for the current task using its registry name.
            entry = classification.get(_registry_name(task) or "")
            if needs_classification:
                # If classification filters are active, apply them to the current task.
                if entry is None:
                    continue  # Skip tasks that lack classification metadata if filters are active.
                if self.category is not None and entry.get("category") != self.category:
                    continue  # Skip if the task's category does not match the specified filter.
                if self.difficulty_level is not None and entry.get("difficulty_level") != self.difficulty_level:
                    continue  # Skip if the task's difficulty level does not match the specified filter.
            # If the task passes all active filters (or no filters are active),
            # enrich it with classification metadata if an entry exists.
            if entry is not None:
                task["category"] = entry.get("category")
                task["difficulty_level"] = entry.get("difficulty_level")
            filtered.append(task)

        return filtered

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve metadata including robust evaluation category and difficulty filters.

        Overrides the parent method to add LIBERO-Plus specific filter information
        to the benchmark's metadata dictionary.

        Returns:
            A metadata dictionary containing base benchmark specifications and
            the active category and difficulty filters.
        """
        meta = super().get_metadata()
        meta["category_filter"] = self.category
        meta["difficulty_filter"] = self.difficulty_level
        return meta
