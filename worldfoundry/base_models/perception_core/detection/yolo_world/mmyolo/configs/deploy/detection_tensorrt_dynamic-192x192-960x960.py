_base_ = ["./base_dynamic.py"]
backend_config = {
    "type": "tensorrt",
    "common_config": {"fp16_mode": False, "max_workspace_size": 1 << 30},
    "model_inputs": [
        {
            "input_shapes": {
                "input": {"min_shape": [1, 3, 192, 192], "opt_shape": [1, 3, 640, 640], "max_shape": [1, 3, 960, 960]}
            }
        }
    ],
}
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # E501
