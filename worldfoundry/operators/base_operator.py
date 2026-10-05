"""Module for the Base operator implementation."""


class BaseOperator(object):
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
        """Interaction template init implementation."""
        if type(self.interaction_template) is not list:
            raise ValueError("interaction_template should be a list")

    def get_interaction(self, interaction):
        """
        utilize this function to update the interaction list
        """
        pass

    def check_interaction(self, interaction):
        """
        utilize this function to check the interaction validity
        """
        pass

    def process_interaction(self):
        """
        utilize this function to process the interaction
        """
        pass

    def process_perception(self):
        """
        utilize this function to process the visual, audio singal
        This function is different from process_interaction for real-time interactive updates
        """
        pass

    def get_interaction_template(self):
        """Get interaction template implementation."""
        return self.interaction_template
    
    def get_interaction_history(self):
        """Get interaction history implementation."""
        return self.interaction_history
    
    def delete_last_interaction(self):
        """Remove the last recorded interaction from the current list.

        Raises:
            ValueError: If there is no recorded interaction to delete.
        """
        if not self.current_interaction:
            raise ValueError("No interaction to delete.")
        self.current_interaction = self.current_interaction[:-1]
