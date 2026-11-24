_base_ = ["./base_static.py"]
onnx_config = {"input_shape": [320, 320], "output_names": ["feat0", "feat1", "feat2"]}
codebase_config = {"model_type": "rknn"}
backend_config = {
    "type": "rknn",
    "common_config": {"target_platform": "rv1126", "optimization_level": 1},
    "quantization_config": {"do_quantization": True, "dataset": None},
    "input_size_list": [[3, 320, 320]],
}
