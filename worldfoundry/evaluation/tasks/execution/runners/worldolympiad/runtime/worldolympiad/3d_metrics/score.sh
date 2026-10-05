python 3d_metrics/score_video_3d.py \
    --video data/6.mp4 \
    --prompt "Stones rolled down the slope" \
    --model-name ./weights/da3 \
    --vlm-backend local \
    --scoring-model qwen/qwen3.5-9b \
    --camera-trajectory ./data/6_da3_camera_trajectory.json \
    --num-workers 1 \
    --gpus 2


python 3d_metrics/score_video_3d.py \
    --video data/real/motor/motor_gt.mp4 \
    --model-name ./weights/da3 \
    --vlm-backend local \
    --scoring-model qwen/qwen3.5-9b \
    --num-workers 1 \
    --gpus 2


ffmpeg -y -i batch_0_gs_video.mp4 -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart -an output_fixed.mp4
