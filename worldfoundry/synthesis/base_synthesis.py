"""
This module defines the `BaseSynthesis` abstract base class, which serves as a common interface
for different synthesis backends. It provides core methods that all synthesis implementations
should adhere to, ensuring a consistent contract for operations like model loading, API initialization,
and prediction.

It also includes a compatibility layer for PyTorch, allowing the module to be imported
and used even in environments where PyTorch is not installed.
"""
from abc import ABC
from functools import wraps


def _lazy_no_grad(func):
    """Retain Torch's decorator semantics without loading Torch for discovery."""
    @wraps(func)
    def predict_without_grad(*args, **kwargs):
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            return func(*args, **kwargs)
        return torch.no_grad()(func)(*args, **kwargs)

    return predict_without_grad

class BaseSynthesis(ABC):
    """
    Abstract Base Class (ABC) defining the fundamental contract for various synthesis backends.

    This class establishes a common interface for different implementations, such as
    local model inference or cloud API-based synthesis, by specifying methods
    that must be implemented by concrete subclasses.
    """

    def __init__(self):
        """
        Initializes the BaseSynthesis instance.

        While the base class constructor is empty, subclasses may implement
        their own initialization logic.
        """
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        """
        Loads a pre-trained synthesis model or configuration.

        This is a class method intended to instantiate a concrete synthesis backend
        from a specified path or identifier. Subclasses must provide an implementation
        for loading their specific models or setting up their backend.

        Args:
            pretrained_model_path (str): The path or identifier for the pre-trained model/backend.
            args (object): An object containing additional arguments for model loading or configuration.
            device (str, optional): The device to load the model onto (e.g., 'cpu', 'cuda'). Defaults to None.
            **kwargs: Additional keyword arguments specific to the subclass implementation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{cls.__name__}.from_pretrained() must be implemented by subclasses.")

    def api_init(self, api_key, endpoint):
        """
        Initializes an API-based synthesis backend with authentication credentials.

        This method is intended for subclasses that interact with external APIs
        for synthesis. It configures the backend with the necessary API key and endpoint.

        Args:
            api_key (str): The API key or token required for authentication with the synthesis service.
            endpoint (str): The URL endpoint for the synthesis service API.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{type(self).__name__}.api_init() must be implemented by subclasses.")

    @_lazy_no_grad
    def predict(self):
        """
        Performs the synthesis prediction using the configured backend.

        This method is decorated with `@_lazy_no_grad` to indicate that
        gradient calculations are not required during prediction, which can save
        memory and improve performance for PyTorch-based implementations.
        Subclasses must implement the actual prediction logic.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{type(self).__name__}.predict() must be implemented by subclasses.")
