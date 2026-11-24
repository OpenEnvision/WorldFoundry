import peft
import torch


def configure_lora_for_model(transformer, model_name, lora_config, is_main_process=True):
    target_linear_modules = set()
    if model_name == "generator":
        adapter_target_modules = ["CausalWanAttentionBlock"]
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)
    target_linear_modules = list(target_linear_modules)
    if is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        if getattr(lora_config, "verbose", False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")
    adapter_type = lora_config.get("type", "lora")
    if adapter_type == "lora":
        peft_config = peft.LoraConfig(
            r=lora_config.get("rank", 16),
            lora_alpha=lora_config.get("alpha", None) or lora_config.get("rank", 16),
            lora_dropout=lora_config.get("dropout", 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")
    lora_model = peft.get_peft_model(transformer, peft_config)
    if is_main_process:
        print("peft_config", peft_config)
        lora_model.print_trainable_parameters()
    return lora_model
