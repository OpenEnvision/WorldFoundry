# `mllm/modelscope_swift`

Inference-only subset of ModelScope Swift (`swift/`): model/tokenizer registry, chat templates,
and LoRA adapter loading. It is a framework, not a model.

## Used by

| Consumer | Purpose |
| --- | --- |
| [`instance_anomaly_detector`](../instance_anomaly_detector/) | Loads the base MLLM and applies the anomaly-detection adapter for VBench-2.0 Instance Preservation |

## Usage

Not called by benchmarks directly. Code that needs it imports
`worldfoundry.base_models.llm_mllm_core.mllm.modelscope_swift.swift.llm` (`get_model_tokenizer`)
and `...swift.tuners.Swift`.
