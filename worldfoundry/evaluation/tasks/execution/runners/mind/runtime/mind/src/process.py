import lpips
import torch
from pyiqa.archs.musiq_arch import MUSIQ
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils.utils import load_time_from_json, print_gpu_memory, get_musiq_spaq_path, get_vitl_path, get_aes_path, extract_actions_from_json, ensure_all_models_downloaded, CACHE_DIR, VideoStreamReader, get_video_length
from utils.dino_utils import load_dinov3_model, extract_dinov3_features
from tqdm import tqdm
import threading
import torch.nn.functional as F
import json
import clip
import os
from pathlib import Path
import torch.multiprocessing as mp
import warnings
from metrics.action import action_accuracy_metric
from metrics.lcm import lcm_metric, merge_lcm_results
from metrics.visual_quality import visual_quality_metric, merge_visual_results
from metrics.dino import dino_mse_metric, merge_dino_results
import time
from collections import namedtuple

Task = namedtuple('Task', ['data', 'retry_count'])

def compute_metrics_single_gpu(task_queue, result_list, gt_root, test_root, dino_path,
                               requested_metrics, video_max_time, process_batch_size, device, gpu_id, stop_event):
    import warnings
    warnings.filterwarnings("ignore", message=".*pretrained.*")
    warnings.filterwarnings("ignore", message=".*Weights.*")
    warnings.filterwarnings("ignore", message=".*video.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*timm\\.models\\.layers.*")
    warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
    """
    在单个GPU上从任务队列获取并处理任务
    """
    # 初始化模型
    if 'lcm' in requested_metrics or 'gsc' in requested_metrics:
        tqdm.write(f"GPU[{gpu_id}]: loading lcm model")
        lpips_metric = lpips.LPIPS(net='alex', spatial=False).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1, 2, 3]).to(device)

    if 'visual' in requested_metrics:
        tqdm.write(f"GPU[{gpu_id}]: loading visual quality model")
        imaging_model = MUSIQ(pretrained_model_path=get_musiq_spaq_path())
        aesthetic_model = torch.nn.Linear(768, 1)
        clip_model, preprocess = clip.load(get_vitl_path(), device=device)

        s = torch.load(get_aes_path(), weights_only=False)
        aesthetic_model.load_state_dict(s)
        aesthetic_model.to(device)
        aesthetic_model.eval()

        imaging_model.to(device)
        imaging_model.training = False

    # 加载DINOv3模型
    if 'dino' in requested_metrics:
        tqdm.write(f"GPU[{gpu_id}]: loading dinov3 model")
        dino_model, dino_processor = load_dinov3_model(dino_path, device)

    tqdm.write(f"GPU[{gpu_id}]: loaded all models")
    processed_count = 0
    try:
        while True:
            # 检查是否需要停止
            if stop_event.is_set():
                tqdm.write(f"GPU{gpu_id}: Stop event received, exiting...")
                break

            try:
                task = task_queue.get(timeout=5)
                data = task.data
                retry_count = task.retry_count
            except:
                continue

            data_path = data['path']
            gt_dir = os.path.join(gt_root, data['perspective'], 'test', data['test_type'])
            test_dir = os.path.join(test_root, data['perspective'], data['test_type'])

            prefix = f"[GPU{gpu_id}] {data_path}"
            try:
                if data['test_type'] == 'mirror_test':
                    if 'gsc' not in requested_metrics:
                        continue
                    result = data.copy()
                    path_videos = [v for v in os.listdir(os.path.join(test_dir, data_path)) if v.endswith('.mp4')]
                    result['video_results'] = []
                    result['videos'] = path_videos
                    result['error'] = None
                    for video in path_videos:
                        vid_result = {'video_name': video, 'error': None}
                        sample_frames = get_video_length(os.path.join(test_dir, data_path, video))
                        vid_result['mark_time'] = sample_frames // 2
                        vid_result['sample_frames'] = sample_frames
                        if sample_frames%2 == 1:
                            vid_result['error'] = 'frame count is not an even number!'
                            tqdm.write(f"{prefix}: Task failed because frame count is not an even number!")
                            result['video_results'].append(vid_result)
                            continue

                        tqdm.write(f"{prefix}: [1/2] Reading videos...")
                        sample_reader = VideoStreamReader(os.path.join(test_dir, data_path, video), start_frame=0, total_frames=sample_frames)
                        _, imgs = sample_reader.read_batch(sample_frames)

                        origin_pred = imgs[:sample_frames // 2]
                        mirror_pred = torch.flip(imgs[sample_frames // 2:], dims=[0])

                        tqdm.write(f"{prefix}: part len: {len(origin_pred)}, {len(mirror_pred)}")
                        
                        tqdm.write(f"{prefix}: [2/2] Computing GSC metrics (MSE/PSNR/SSIM/LPIPS)...")
                        gsc = lcm_metric(origin_pred, mirror_pred, lpips_metric, ssim_metric, psnr_metric, process_batch_size, device)
                        vid_result['gsc'] = gsc
                        result['video_results'].append(vid_result)

                        del sample_reader                                                                                                                                                                  
                        torch.cuda.empty_cache()

                else:
                    mark_time, total_time = load_time_from_json(os.path.join(gt_dir, data_path, 'action.json'))
                    result = data.copy()
                    result['error'] = None
                    result['mark_time'] = mark_time
                    result['total_time'] = total_time

                    gt_frames = get_video_length(os.path.join(gt_dir, data_path, 'video.mp4'))
                    sample_frames = get_video_length(os.path.join(test_dir, data_path, 'video.mp4'))
                    tqdm.write(f"{prefix}: gt_frames={gt_frames}, sample={sample_frames}, total_time={total_time}, mark_time={mark_time}")
                    result['sample_frames'] = sample_frames

                    if sample_frames != total_time - mark_time:
                        real_time = min(total_time - mark_time, sample_frames)
                    else: real_time = sample_frames

                    if video_max_time is not None:
                        real_time = min(real_time, video_max_time)

                    gt_reader = VideoStreamReader(os.path.join(gt_dir, data_path, 'video.mp4'), start_frame=mark_time, total_frames=mark_time + real_time)
                    sample_reader = VideoStreamReader(os.path.join(test_dir, data_path, 'video.mp4'), start_frame=0, total_frames=real_time)

                    while True:
                        tqdm.write(f"{prefix}: [1/5] Reading videos...")
                        is_ended, gt_imgs = gt_reader.read_batch(process_batch_size * 10)
                        _, sample_imgs = sample_reader.read_batch(process_batch_size * 10)

                        if is_ended or gt_imgs is None or sample_imgs is None:
                            break

                        if 'lcm' in requested_metrics:
                            tqdm.write(f"{prefix}: [2/5] Computing LCM metrics (MSE/PSNR/SSIM/LPIPS)...")
                            lcm = lcm_metric(sample_imgs, gt_imgs, lpips_metric, ssim_metric, psnr_metric, process_batch_size, device)
                            result['lcm'] = merge_lcm_results(result.get('lcm'), lcm)

                        if 'visual' in requested_metrics:
                            tqdm.write(f"{prefix}: [3/5] Computing visual quality metrics...")
                            vq = visual_quality_metric(sample_imgs, imaging_model, aesthetic_model, clip_model, process_batch_size, device)
                            result['visual_quality'] = merge_visual_results(result.get('visual_quality'), vq)

                        if 'dino' in requested_metrics:
                            tqdm.write(f"{prefix}: [4/5] Computing dino metrics...")
                            dino = dino_mse_metric(sample_imgs, gt_imgs, dino_model, dino_processor, device, process_batch_size)
                            result['dino'] = merge_dino_results(result.get('dino'), dino)

                        del gt_imgs, sample_imgs

                    del gt_reader, sample_reader
                    torch.cuda.empty_cache()

                    if 'action' in requested_metrics:
                        tqdm.write(f"{prefix}: [4/5] Computing action accuracy (ViPE)...")
                        # 计算action accuracy
                        actions = extract_actions_from_json(os.path.join(gt_dir, data_path, 'action.json'), mark_time, 97)
                        action = action_accuracy_metric(
                            os.path.join(test_dir, data_path, 'video.mp4'),
                            os.path.join(gt_dir, data_path, 'video.mp4'),
                            mark_time,
                            actions,
                            max_frames = 97,
                            gt_data_dir = os.path.join(gt_dir, data_path),
                            verbose_prefix=f"  {prefix}",
                            gpu_id=gpu_id
                        )
                        if action is not None:
                            result['action'] = action

                    tqdm.write(f"{prefix}: Finish Task!")

            except KeyboardInterrupt:
                tqdm.write(f"{prefix}: Interrupted by user. Exiting...")
                if stop_event is not None:
                    stop_event.set()
                raise
            except Exception as e:
                result['error'] = str(e)
                if retry_count < 3:
                    tqdm.write(f"{prefix}: Error processing {data_path} on GPU{gpu_id}: {e}, retrying ({retry_count + 1}/3)")
                    task_queue.put(Task(data, retry_count + 1))
                else:
                    tqdm.write(f"{prefix}: Error processing {data_path} on GPU{gpu_id}: {e}, max retries exceeded, giving up")
                    result_list.append(result)
                continue

            result_list.append(result)
            processed_count += 1

    finally:
        tqdm.write(f"GPU{gpu_id}: Cleaning up...")
        if 'lcm' in requested_metrics or 'gsc' in requested_metrics:
            del lpips_metric, ssim_metric, psnr_metric
        if 'visual' in requested_metrics:
            del imaging_model, aesthetic_model, clip_model
        torch.cuda.empty_cache()

def compute_metrics(gt_root, test_root, dino_path, output_path, requested_metrics=['lcm', 'visual', 'dino', 'action'],
                   video_max_time=100, process_batch_size=10, num_gpus=1):
    result_dict = {'data':[], 'video_max_time':video_max_time}

    mp.set_start_method('spawn', force=True)

    all_data = []
    for perspective in ['1st_data', '3rd_data']:
        for test_type in ['mem_test', 'action_space_test', 'mirror_test']:
            if test_type == 'mirror_test':
                if 'gsc' in requested_metrics:
                    test_dir = os.path.join(test_root, perspective, test_type)
                    all_data += [{'path': d, 'perspective': perspective, 'test_type': test_type }
                        for d in os.listdir(test_dir)]
            else:
                if 'lcm' in requested_metrics or 'visual' in requested_metrics or 'dino' in requested_metrics or 'action' in requested_metrics:
                    gt_dir = os.path.join(gt_root, perspective, 'test', test_type)
                    test_dir = os.path.join(test_root, perspective, test_type)

                    all_data += [{'path': d, 'perspective': perspective, 'test_type': test_type}
                        for d in os.listdir(test_dir) if os.path.exists(os.path.join(gt_dir, d))]

    if len(all_data) == 0:
        tqdm.write(f"No data found!")
        return

    total_tasks = len(all_data)
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Total data: {total_tasks}, Using {num_gpus} GPU(s) with task queue")
    tqdm.write(f"{'='*60}\n")

    ensure_all_models_downloaded()

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_list = manager.list()
    stop_event = manager.Event()

    for task in all_data:
        task_queue.put(Task(task, 0))

    # 进度监控线程
    def monitor_progress():
        pbar = tqdm(total=total_tasks, desc="Progress", unit="video")
        last_count = 0
        try:
            while not stop_event.is_set() or len(result_list) < total_tasks:
                current_count = len(result_list)
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                    with open(output_path, 'w') as f:
                        result_dict['data'] = list(result_list)
                        json.dump(result_dict, f, indent=2)
                if last_count >= total_tasks:
                    break
                time.sleep(0.5)
            stop_event.set()
        except (ConnectionResetError, OSError):
            pass  # Ctrl-C时manager连接关闭，忽略
        finally:
            pbar.close()

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    try:
        with mp.get_context("spawn").Pool(processes=num_gpus) as pool:
            worker_args = [
                (task_queue, result_list, gt_root, test_root, dino_path,
                    requested_metrics, video_max_time, process_batch_size, f'cuda:{i}', i, stop_event)
                for i in range(num_gpus)
            ]

            pool.starmap(compute_metrics_single_gpu, worker_args)

        monitor_thread.join()
        result_dict['data'] = list(result_list)

    except KeyboardInterrupt:
        tqdm.write("\n\nInterrupted by user! Merging partial results...")
        stop_event.set()
        pool.terminate()
        pool.join()
        monitor_thread.join(timeout=2)
        result_dict['data'] = list(result_list)
        tqdm.write(f"  Merged {len(result_dict['data'])} results from shared list")

    return result_dict

if __name__ == '__main__':
    import argparse
    warnings.filterwarnings("ignore", message=".*pretrained.*")
    warnings.filterwarnings("ignore", message=".*Weights.*")
    warnings.filterwarnings("ignore", message=".*video.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*timm.models.layers.*")
    warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

    parser = argparse.ArgumentParser(description='Compute video metrics with multi-GPU support')
    parser.add_argument('--gt_root', type=str, required=True, help='Ground truth data root directory')
    parser.add_argument('--test_root', type=str, required=True, help='Test data root directory')
    parser.add_argument('--dino_path', type=str, default='./dinov3_vitb16',
                       help='dinov3 weight directory, for example ./dinov3_vitb16')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--video_max_time', type=int, default=None, help='Maximum video frames (default: None = use all frames)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--metrics', type=str, default='lcm,visual,dino,action,gsc', help='Requested metrics to compute, comma separated (e.g. dino,visual)')

    args = parser.parse_args()

    # 检查GPU数量
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        tqdm.write(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        args.num_gpus = available_gpus

    # 准备输出路径
    if args.output:
        output_path = args.output
    else:
        from datetime import datetime
        output_path = f'result_{os.path.basename(args.test_root)}_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.json'

    tqdm.write(f"Starting computation with {args.num_gpus} GPU(s)...")
    tqdm.write(f"Output will be saved to: {output_path}")

    video_results = None
    try:
        video_results = compute_metrics(
            args.gt_root,
            args.test_root,
            args.dino_path,
            output_path,
            requested_metrics=[m.strip() for m in args.metrics.split(',')],
            video_max_time=args.video_max_time,
            num_gpus=args.num_gpus
        )
    except KeyboardInterrupt:
        tqdm.write("\n\n" + "="*60)
        tqdm.write("INTERRUPTED BY USER (Ctrl+C)")
        tqdm.write("="*60)
        if video_results is None:
            tqdm.write("No results to save (computation was interrupted too early)")

    finally:
        # 保存结果(即使被中断)
        if video_results is not None:
            tqdm.write(f"{len(video_results['data'])} videos are processed")
            tqdm.write(f"checking output path {output_path}:")
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    saved_results = json.load(f)
                    tqdm.write(f"{len(saved_results['data'])} videos are saved")
        else:
            tqdm.write("\nNo results to save.")
    