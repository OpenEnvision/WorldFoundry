_base_ = ["./base_static.py"]
onnx_config = {"input_shape": (640, 640)}
backend_config = {
    "type": "tensorrt",
    "common_config": {"fp16_mode": True, "max_workspace_size": 1 << 30, "int8_mode": True},
    "model_inputs": [
        {
            "input_shapes": {
                "input": {"min_shape": [1, 3, 640, 640], "opt_shape": [1, 3, 640, 640], "max_shape": [1, 3, 640, 640]}
            }
        }
    ],
}
calib_config = {"create_calib": True, "calib_file": "calib_data.h5"}
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # E501
