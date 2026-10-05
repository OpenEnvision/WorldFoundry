from worldfoundry.pipelines.hunyuan_world.pipeline_hy_world_2p0 import HYWorld2Pipeline


INPUT_PATH = "./worldfoundry/data/test_cases/test_image_seq_case1"
OUTPUT_PATH = "./output/hy_world_2p0"


pipeline = HYWorld2Pipeline.from_pretrained(
    model_path="tencent/HY-World-2.0",
    device="cuda",
)

result_dir = pipeline(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
)

print("HY-World-2.0 reconstruction completed.")
print(f"Results saved to: {result_dir}")
