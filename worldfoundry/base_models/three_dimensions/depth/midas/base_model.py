import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"), weights_only=True)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        parameters = {
            key: value for key, value in parameters.items() if not key.endswith(".attn.relative_position_index")
        }
        self.load_state_dict(parameters)
