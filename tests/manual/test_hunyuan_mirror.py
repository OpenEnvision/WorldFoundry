import os
from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_mirror import HunyuanMirrorPipeline

# set input and output paths
input_path = "./worldfoundry/data/test_cases/test_image_seq_case1"
output_path = "output/hunyuan_mirror_mirror"

image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
image_paths = []
for ext in image_extensions:
    image_paths.extend([os.path.join(input_path , f) for f in os.listdir(input_path )
                        if f.lower().endswith(ext)])
if not image_paths:
    print(f"❌ 目录中没有找到图片文件: {input_path }")
    exit(1)

# load model
pipeline = HunyuanMirrorPipeline.from_pretrained(
    model_path="tencent/HunyuanWorld-Mirror",
    output_path=output_path,
    device="cuda"
)

# inference
processing_results = pipeline(
    image_path=image_paths,
    # the input image_path contains multiple images, may be need another parameter name.
    output_path=output_path
)

# save results
results = pipeline.save_results(
    results=processing_results,
    save_pointmap=True,
    save_depth=True,
    save_normal=True
)

print("3D重建完成！")
print(f"结果已保存到: {output_path}")
