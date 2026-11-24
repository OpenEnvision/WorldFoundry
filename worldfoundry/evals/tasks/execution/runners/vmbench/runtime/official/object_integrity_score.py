import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine

# save original register
original_register = mmengine.registry.Registry.register_module


def safe_register(self, *args, **kwargs):
    try:
        return original_register(self, *args, **kwargs)
    except KeyError:
        # if register failed, return None
        return None


# replace register method
mmengine.registry.Registry.register_module = safe_register

# Provider imports require the MMEngine registry policy above.
import numpy as np  # noqa: E402
from mmpose.apis import inference_topdown  # noqa: E402 - Provider imports require the MMEngine registry policy above.
from mmpose.apis import (  # noqa: E402 - Provider imports require the MMEngine registry policy above.
    init_model as init_pose_estimator,
)
from mmpose.evaluation.functional import (  # noqa: E402 - Provider imports require the MMEngine registry policy above.
    nms,
)
from mmpose.registry import VISUALIZERS  # noqa: E402 - Provider imports require the MMEngine registry policy above.
from mmpose.structures import (  # noqa: E402 - Provider imports require the MMEngine registry policy above.
    merge_data_samples,
    split_instances,
)
from mmpose.utils import (  # noqa: E402 - Provider imports require the MMEngine registry policy above.
    adapt_mmdet_pipeline,
)

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from bench_utils.pose_utils import (  # noqa: E402 - Follows the optional import block.
    analyze_joint_angles,
    analyze_lengths_over_time,
)
from tqdm import tqdm  # noqa: E402 - Follows the optional import block.

RUNTIME_ROOT = os.path.dirname(__file__)


def process_one_image(args, img, detector, pose_estimator, visualizer=None, show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id, pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order="rgb")
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            "result",
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr,
        )

    # if there is no instance detected, return None
    return data_samples.get("pred_instances", None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--det_config",
        default=os.path.join(RUNTIME_ROOT, "mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"),
        help="Config file for detection",
    )
    parser.add_argument(
        "--det_checkpoint",
        default="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
        help="Checkpoint file for detection",
    )
    parser.add_argument(
        "--pose_config",
        default=os.path.join(
            RUNTIME_ROOT, "mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
        ),
        help="Config file for pose",
    )
    parser.add_argument(
        "--pose_checkpoint",
        default="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
        help="Checkpoint file for pose",
    )
    parser.add_argument("--meta-info-path", default="meta_info.json", help="Video meta info path")
    parser.add_argument("--input", type=str, default="", help="Image/Video file")
    parser.add_argument("--show", action="store_true", default=False, help="whether to show img")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="root of the output img file. Default not saving the visualization images.",
    )
    parser.add_argument(
        "--save-predictions", action="store_true", default=False, help="whether to save predicted results"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--det-cat-id", type=int, default=0, help="Category id for bounding box detection model")
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Bounding box score threshold")
    parser.add_argument("--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds")
    parser.add_argument(
        "--draw-heatmap", action="store_true", default=False, help="Draw heatmap predicted by the model"
    )
    parser.add_argument(
        "--show-kpt-idx", action="store_true", default=False, help="Whether to show the index of keypoints"
    )
    parser.add_argument(
        "--skeleton-style", default="mmpose", type=str, choices=["mmpose", "openpose"], help="Skeleton style selection"
    )
    parser.add_argument("--radius", type=int, default=3, help="Keypoint radius for visualization")
    parser.add_argument("--thickness", type=int, default=1, help="Link thickness for visualization")
    parser.add_argument("--show-interval", type=int, default=0, help="Sleep seconds per frame")
    parser.add_argument("--alpha", type=float, default=0.8, help="The transparency of bboxes")
    parser.add_argument("--draw-bbox", action="store_true", help="Draw bboxes of instances")

    assert has_mmdet, "Please install mmdet to run the demo."

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # build detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options={"model": {"test_cfg": {"output_heatmaps": args.draw_heatmap}}},
    )

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    with open(args.meta_info_path) as f:
        meta_infos = json.load(f)

    for meta_info in tqdm(meta_infos):
        args.input = meta_info["filepath"]

        output_file = None
        if args.output_root:
            mmengine.mkdir_or_exist(args.output_root)
            output_file = os.path.join(args.output_root, os.path.basename(args.input))
            if args.input == "webcam":
                output_file += ".mp4"

        input_type = "webcam" if args.input == "webcam" else mimetypes.guess_type(args.input)[0].split("/")[0]

        if input_type == "image":
            # inference
            pred_instances = process_one_image(args, args.input, detector, pose_estimator, visualizer)

            if args.save_predictions:
                pred_instances_list = split_instances(pred_instances)

            if output_file:
                img_vis = visualizer.get_image()
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

        elif input_type in ["webcam", "video"]:
            cap = cv2.VideoCapture(0) if args.input == "webcam" else cv2.VideoCapture(args.input)

            video_writer = None
            pred_instances_list = []
            frame_idx = 0

            while cap.isOpened():
                success, frame = cap.read()
                frame_idx += 1

                if not success:
                    break

                # topdown pose estimation
                pred_instances = process_one_image(args, frame, detector, pose_estimator, visualizer, 0.001)

                if args.save_predictions:
                    # save prediction results
                    pred_instances_list.append({"frame_id": frame_idx, "instances": split_instances(pred_instances)})

                # output videos
                if output_file:
                    frame_vis = visualizer.get_image()

                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        # the size of the image with visualization may vary
                        # depending on the presence of heatmaps
                        video_writer = cv2.VideoWriter(
                            output_file,
                            fourcc,
                            12,  # saved fps
                            (frame_vis.shape[1], frame_vis.shape[0]),
                        )

                    video_writer.write(mmcv.rgb2bgr(frame_vis))

                if args.show:
                    # press ESC to exit
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

                    time.sleep(args.show_interval)

            if video_writer:
                video_writer.release()

            cap.release()

        else:
            args.save_predictions = False
            raise ValueError(f"file {os.path.basename(args.input)} has invalid format.")

        if args.save_predictions:
            # Analyze if there are any abnormalities in component length changes
            _, length_score = analyze_lengths_over_time(pred_instances_list)
            # Analyze if there are any abnormalities in component angle changes
            _, angle_score = analyze_joint_angles(pred_instances_list)
            score = length_score * 0.5 + angle_score * 0.5
            meta_info["object_integrity_score"] = score
            with open(args.meta_info_path, "w") as f:
                json.dump(meta_infos, f, indent=4)


if __name__ == "__main__":
    main()
