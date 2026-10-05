import json
import os
import subprocess

from worldfoundry.runtime.jobs import run_bounded_command


def get_video_info(filepath, *, timeout_seconds: int = 30):
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,duration",
        "-of", "json", filepath
    ]
    result = run_bounded_command(cmd, timeout=timeout_seconds)
    if result["timed_out"]:
        raise TimeoutError(f"ffprobe timed out after {timeout_seconds}s for {filepath}")
    if result["returncode"] != 0:
        detail = str(result["stderr"]).strip()
        raise RuntimeError(
            f"ffprobe failed for {filepath} with code {result['returncode']}: {detail}"
        )
    info = json.loads(result["stdout"])
    duration = float(info['streams'][0]['duration'])
    fr_str = info['streams'][0]['avg_frame_rate']
    if '/' in fr_str:
        num, den = map(int, fr_str.split('/'))
        fps = num / den if den != 0 else 25
    else:
        fps = float(fr_str)
    return duration, fps

def split_video(filepath, output_folder):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    duration, fps = get_video_info(filepath)
    total_segments = int(duration) - 1 

    for i in range(total_segments): 
        out_path = os.path.join(output_folder, f"{name}_clip_{i:04d}.mp4")
        if os.path.exists(out_path):
            continue
        cmd = [
            "ffmpeg", "-y",
            "-i", filepath,
            "-ss", str(i),       
            "-t", "2",          
            "-r", str(fps),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            out_path
        ]
        subprocess.run(cmd)

def split(input_folder, model_name):
    output_folder = f"./model_clip/{model_name}"
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            split_video(os.path.join(input_folder, file), output_folder)
