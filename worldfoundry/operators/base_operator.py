"""Module for the Base operator implementation."""

from typing import Any


class BaseOperator:
    """Operator class for the Base model integration."""

    def __init__(self, operation_types=None):
        """
        operation includes:
            - textual_instruction
            - visual_instruction
            - action_instruction (including mouse and keyboard to
              control the trajectory and viewpoint.)
        """
        if operation_types is None:
            operation_types = []
        self.operation_types = operation_types
        self.interaction_template = []
        self.current_interaction = []
        self.interaction_history = []

    def interaction_template_init(self):
        """Check that ``interaction_template`` is a list.

        Raises:
            ValueError: ``interaction_template`` is not a list.
        """
        if type(self.interaction_template) is not list:
            raise ValueError("interaction_template should be a list")

    def get_interaction(self, *args: Any, **kwargs: Any) -> Any:
        """
        utilize this function to update the interaction list
        """

    def check_interaction(self, *args: Any, **kwargs: Any) -> Any:
        """
        utilize this function to check the interaction validity
        """

    def process_interaction(self, *args: Any, **kwargs: Any) -> Any:
        """
        utilize this function to process the interaction
        """

    def process_perception(self, *args: Any, **kwargs: Any) -> Any:
        """
        utilize this function to process the visual, audio singal
        This function is different from process_interaction for real-time interactive updates
        """

    def get_interaction_template(self):
        """Return the interactions this operator accepts."""
        return self.interaction_template

    def get_interaction_history(self):
        """Return the interactions processed so far."""
        return self.interaction_history

    def delete_last_interaction(self):
        """Remove the last recorded interaction from the current list.

        Raises:
            ValueError: If there is no recorded interaction to delete.
        """
        if not self.current_interaction:
            raise ValueError("No interaction to delete.")
        self.current_interaction = self.current_interaction[:-1]
