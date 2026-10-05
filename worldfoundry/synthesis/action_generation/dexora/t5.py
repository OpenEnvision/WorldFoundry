import torch
from transformers import AutoTokenizer, T5EncoderModel

from worldfoundry.core.execution.device import resolve_inference_dtype


class T5Embedder:
    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=120,
        local_files_only=True,
    ):
        self.device = torch.device(device)
        self.torch_dtype = resolve_inference_dtype(self.device, torch_dtype or "auto")
        self.cache_dir = cache_dir

        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }

            if use_offload_folder is not None:
                t5_model_kwargs["offload_folder"] = use_offload_folder
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder.embed_tokens": self.device,
                    "encoder.block.0": self.device,
                    "encoder.block.1": self.device,
                    "encoder.block.2": self.device,
                    "encoder.block.3": self.device,
                    "encoder.block.4": self.device,
                    "encoder.block.5": self.device,
                    "encoder.block.6": self.device,
                    "encoder.block.7": self.device,
                    "encoder.block.8": self.device,
                    "encoder.block.9": self.device,
                    "encoder.block.10": self.device,
                    "encoder.block.11": self.device,
                    "encoder.block.12": "disk",
                    "encoder.block.13": "disk",
                    "encoder.block.14": "disk",
                    "encoder.block.15": "disk",
                    "encoder.block.16": "disk",
                    "encoder.block.17": "disk",
                    "encoder.block.18": "disk",
                    "encoder.block.19": "disk",
                    "encoder.block.20": "disk",
                    "encoder.block.21": "disk",
                    "encoder.block.22": "disk",
                    "encoder.block.23": "disk",
                    "encoder.final_layer_norm": "disk",
                    "encoder.dropout": "disk",
                }
            else:
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder": self.device,
                }

        else:
            t5_model_kwargs = dict(t5_model_kwargs)
            t5_model_kwargs["torch_dtype"] = self.torch_dtype

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        if from_pretrained is None:
            raise ValueError("A local T5 checkpoint directory is required.")
        from pathlib import Path

        local_root = Path(from_pretrained).expanduser()
        if not local_root.is_dir():
            raise FileNotFoundError(f"Local T5 checkpoint directory not found: {local_root}")
        if not local_files_only:
            raise ValueError("Dexora text loading is local-only")
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_root,
            model_max_length=model_max_length,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = T5EncoderModel.from_pretrained(
            local_root,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=False,
            **t5_model_kwargs,
        ).eval()
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask
