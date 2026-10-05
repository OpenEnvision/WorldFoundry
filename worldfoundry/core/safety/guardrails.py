"""Model-independent safety and postprocessing guardrail orchestration.

This module is interface-level (two Protocols and an orchestration
class), so it deliberately avoids heavyweight imports: logging goes
through the stdlib and numpy is imported only for type checking.
Concrete guardrail implementations (prompt classifiers, NSFW video
filters, face blur) bring their own model dependencies.

:class:`GuardrailRunner` is fail-open when no classifiers are
configured: it logs a warning and reports safe. Deployments that
require enforcement must attach at least one
:class:`ContentSafetyGuardrail`. Postprocessors are applied in order
and may be empty (the original frames are returned).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Protocols — classifiers and frame postprocessors; no model imports here
# ──────────────────────────────────────────────────────────────────────────


class ContentSafetyGuardrail(Protocol):
    """Interface implemented by prompt and media safety classifiers."""

    def is_safe(self, input: Any) -> tuple[bool, str]:
        """Return whether ``input`` is safe and an optional explanation."""
        ...


class PostprocessingGuardrail(Protocol):
    """Interface implemented by safety postprocessors such as face blurring."""

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Return safety-processed frames."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator — fail-open without classifiers; postprocessors may be empty
# ──────────────────────────────────────────────────────────────────────────


class GuardrailRunner:
    """Run reusable safety classifiers and postprocessors in sequence."""

    def __init__(
        self,
        safety_models: Sequence[ContentSafetyGuardrail] | None = None,
        generic_block_msg: str = "",
        generic_safe_msg: str = "",
        postprocessors: Sequence[PostprocessingGuardrail] | None = None,
    ) -> None:
        """Configure classifier and postprocessing chains.

        Args:
            safety_models: Ordered classifiers. Evaluation stops on the first
                unsafe result.
            generic_block_msg: Optional public message replacing classifier
                details when a request is blocked.
            generic_safe_msg: Message returned after all classifiers pass.
            postprocessors: Ordered frame transforms applied by ``postprocess``.
        """
        self.safety_models = safety_models
        self.generic_block_msg = generic_block_msg
        self.generic_safe_msg = generic_safe_msg or "Prompt is safe"
        self.postprocessors = postprocessors

    def run_safety_check(self, input: Any) -> tuple[bool, str]:
        """Run classifiers in order and return the first block or final safe result.

        Note the deliberate fail-open default: with no configured safety
        models the input is reported safe (with a warning). Deployments that
        require enforcement must configure at least one classifier.
        """
        if not self.safety_models:
            log.warning("No safety models found, returning safe")
            return True, self.generic_safe_msg

        for guardrail in self.safety_models:
            guardrail_name = type(guardrail).__name__.upper()
            log.debug("Running guardrail: %s", guardrail_name)
            safe, message = guardrail.is_safe(input)
            if not safe:
                return False, self.generic_block_msg or f"{guardrail_name}: {message}"
        return True, self.generic_safe_msg

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Apply every configured safety postprocessor to a frame array."""
        if not self.postprocessors:
            log.warning("No postprocessors found, returning original frames")
            return frames

        for guardrail in self.postprocessors:
            log.debug("Running guardrail: %s", type(guardrail).__name__.upper())
            frames = guardrail.postprocess(frames)
        return frames
