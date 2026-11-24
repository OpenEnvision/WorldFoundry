"""Convert a PyTorch .pth file to .safetensors format."""

import argparse

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Convert .pth to .safetensors")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pth file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors file")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    state_dict = torch.load(args.input, map_location="cpu", weights_only=True)

    # If the checkpoint wraps the state dict in a key, unwrap it
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    print(f"Saving {args.output} ({len(state_dict)} tensors) ...")
    save_file(state_dict, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
