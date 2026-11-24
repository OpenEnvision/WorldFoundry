_base_ = ["./base_dynamic.py"]
backend_config = {
    "type": "tensorrt",
    "common_config": {"fp16_mode": True, "max_workspace_size": 1 << 32},
    "model_inputs": [
        {
            "input_shapes": {
                "input": {"min_shape": [1, 3, 64, 64], "opt_shape": [1, 3, 640, 640], "max_shape": [1, 3, 1344, 1344]}
            }
        }
    ],
}
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # E501
