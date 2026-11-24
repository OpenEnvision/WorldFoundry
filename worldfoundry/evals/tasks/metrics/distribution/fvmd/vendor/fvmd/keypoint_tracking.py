import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import fvmd.utils.improc
import fvmd.utils.misc
from fvmd.nets.pips2 import Pips

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

PIPS_WEIGHTS = (
    "https://github.com/ljh0v0/FVMD-frechet-video-motion-distance/releases/download/pips2_weights/pips2_weights.pth"
)


def calc_velocity(trajs_e):
    """Take trace from Point tracking algorithm; return velocity of each frames

    Args:
        trajs_e (torch.tensor): (Batch_size, Frames, Points, 2)

    Returns:
        velocity (torch.tensor): (Batch_size, Frames, Points, 2)
    """
    B, S, N, _ = trajs_e.shape
    trajs_e0 = trajs_e[:, :-1]  # B,S-1,N,2
    trajs_e1 = trajs_e[:, 1:]  # B,S-1,N,2
    velocity = trajs_e1 - trajs_e0  # B,S-1,N,2
    velocity = torch.cat([torch.zeros(B, 1, N, 2).cuda(), velocity], dim=1)  # B,S,N,2
    return velocity


def calc_acceleration(velocity):
    """Take Velocity calculated from Point tracking algorithm; return acc of each frames

    Args:
        trajs_e (torch.tensor): (Batch_size, Frames, Points, 2)

    Returns:
        velocity (torch.tensor): (Batch_size, Frames, Points, 2)
    """
    B, S, N, _ = velocity.shape
    velocity0 = velocity[:, 1:-1]  # B,S-2,N,2
    velocity1 = velocity[:, 2:]  # B,S-2,N,2
    acceleration = velocity1 - velocity0  # B,S-2,N,2
    acceleration = torch.cat(
        [torch.zeros(B, 1, N, 2).cuda(), torch.zeros(B, 1, N, 2).cuda(), acceleration], dim=1
    )  # B,S,N,2
    return acceleration


def run_tracking(model, rgbs, N=64, iters=16):
    rgbs = rgbs.cuda().float()  # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert B == 1

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = fvmd.utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device="cuda")
    grid_y = 8 + grid_y.reshape(B, -1) / float(N_ - 1) * (H - 16)
    grid_x = 8 + grid_x.reshape(B, -1) / float(N_ - 1) * (W - 16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1)  # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1, S, 1, 1)

    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]

    return trajs_e


def tracking_fullseq(model, rgbs, sw, N=400, iters=8, S_max=16, name="sample"):
    rgbs = rgbs.cuda().float()  # B,S,C,H,W

    B, S, C, H, W = rgbs.shape
    assert B == 1
    # print('this video is %d frames long' % S)

    # zero-vel init

    cur_frame = 0
    done = False
    trajs = []
    velo = []
    acc = []
    while not done:
        end_frame = cur_frame + S_max

        if end_frame > S:
            break
        # print('working on subseq %d:%d' % (cur_frame, end_frame))

        rgb_seq = rgbs[:, cur_frame:end_frame]

        trajs_e = run_tracking(model, rgb_seq, N=N, iters=iters)

        velocity = calc_velocity(trajs_e)
        acceleration = calc_acceleration(trajs_e)
        velo.append(velocity)
        acc.append(acceleration)
        trajs.append(trajs_e)

        # TODO: delete this
        if sw is not None and sw.save_this and cur_frame == 0:
            sw.summ_traj2ds_on_rgbs(
                "outputs/" + name,
                trajs_e[0:1],
                fvmd.utils.improc.preprocess_color(rgb_seq[0:1]),
                cmap="hot",
                linewidth=1,
                show_dots=False,
            )

        cur_frame = cur_frame + S_max - 1

    return trajs, velo, acc


def track_keypoints(
    log_dir,
    gen_dataset,
    gt_dataset,
    v_stride=1,
    B=1,  # batchsize
    S=16,  # seqlen
    N=400,  # number of points per clip
    stride=8,  # spatial stride of the model
    iters=16,  # inference steps of the model
    image_size=(256, 256),  # input resolution
    shuffle=False,  # dataset shuffling
    init_dir=PIPS_WEIGHTS,
    device_ids=[0],  # noqa: B006 - Read-only default; retain the public signature.
):
    device = "cuda:%d" % device_ids[0]  # noqa: UP031 - Caller-provided device id; keep the printf integer conversion.

    assert B == 1  # B>1 not implemented here
    assert image_size[0] % 32 == 0
    assert image_size[1] % 32 == 0

    # autogen a descriptive name
    model_name = "keypoints_tracking"
    writer_x = SummaryWriter(log_dir + "/" + model_name, max_queue=10, flush_secs=60)

    model = Pips(stride=stride).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    fvmd.utils.misc.count_parameters(model)

    # _ = saverloader.load(init_dir, model.module)
    state_dict = load_state_dict_from_url(PIPS_WEIGHTS, progress=True)
    model.module.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    dataset_size = len(gen_dataset)
    log_freq = dataset_size / 2

    dataloader_gen = DataLoader(gen_dataset, batch_size=1, shuffle=shuffle, num_workers=1)
    iterloader_gen = iter(dataloader_gen)

    dataloader_gt = DataLoader(gt_dataset, batch_size=1, shuffle=shuffle, num_workers=1)
    iterloader_gt = iter(dataloader_gt)

    global_step = 0
    all_traj_gen = []
    all_velo_gen = []
    all_acc_gen = []
    all_traj_gt = []
    all_velo_gt = []
    all_acc_gt = []
    while global_step < dataset_size:
        global_step += 1
        iter_start_time = time.time()
        with torch.no_grad():
            torch.cuda.empty_cache()
        sw_x = fvmd.utils.improc.Summ_writer(
            writer=writer_x, global_step=global_step, log_freq=log_freq, fps=min(S, 8), scalar_freq=1, just_gif=True
        )
        try:
            sample_gen = next(iterloader_gen)
            sample_gt = next(iterloader_gt)
        except StopIteration:
            iterloader_gen = iter(dataloader_gen)
            sample_gen = next(iterloader_gen)
            iterloader_gt = iter(dataloader_gt)
            sample_gt = next(iterloader_gt)
        iter_rtime = time.time() - iter_start_time
        with torch.no_grad():
            trajs_gen, velo_gen, acc_gen = tracking_fullseq(
                model, sample_gen, sw_x, N=N, iters=iters, S_max=S, name=f"{global_step:04d}_sample_gen"
            )
            trajs_gt, velo_gt, acc_gt = tracking_fullseq(
                model, sample_gt, sw_x, N=N, iters=iters, S_max=S, name=f"{global_step:04d}_sample_gt"
            )

            all_traj_gen.extend(trajs_gen)
            all_velo_gen.extend(velo_gen)
            all_acc_gen.extend(acc_gen)
            all_traj_gt.extend(trajs_gt)
            all_velo_gt.extend(velo_gt)
            all_acc_gt.extend(acc_gt)

        iter_itime = time.time() - iter_start_time

        logger.info(
            f"{model_name}; step {global_step:06d}/{dataset_size}; rtime {iter_rtime:.2f}; itime {iter_itime:.2f}"
        )

    # save velo & acc
    all_traj_gen = torch.concatenate(all_traj_gen, dim=0).cpu().numpy()
    np.save(os.path.join(log_dir, "trajctorys_gen.npy"), all_traj_gen)
    all_velo_gen = torch.concatenate(all_velo_gen, dim=0).cpu().numpy()
    np.save(os.path.join(log_dir, "velocity_gen.npy"), all_velo_gen)
    all_acc_gen = torch.concatenate(all_acc_gen, dim=0).cpu().numpy()
    np.save(os.path.join(log_dir, "acceleration_gen.npy"), all_acc_gen)

    all_traj_gt = torch.concatenate(all_traj_gt, dim=0).cpu().numpy()
    np.save(os.path.join(log_dir, "trajctorys_gt.npy"), all_traj_gt)
    all_velo_gt = torch.concatenate(all_velo_gt, dim=0).cpu().numpy()
    np.save(os.path.join(log_dir, "velocity_gt.npy"), all_velo_gt)
    all_acc_gt = torch.concatenate(all_acc_gt, dim=0).cpu().numpy()
    np.save(os.path.join(log_dir, "acceleration_gt.npy"), all_acc_gt)

    logger.info(f"velo_gen shape: {all_velo_gen.shape!s}")

    return all_velo_gen, all_velo_gt, all_acc_gen, all_acc_gt


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_dir", type=str, help=("Path to the log directory"))
    parser.add_argument("--dataset_gen", type=str, help=("Path to the generated video directory"))
    parser.add_argument("--dataset_gt", type=str, help=("Path to the ground truth video directory"))
    parser.add_argument("--v_stride", type=int, default=1, help=("Stride of the video"))
    args = parser.parse_args()

    track_keypoints(
        log_dir=args.log_dir, dataset_gen=args.dataset_gen, dataset_gt=args.dataset_gt, v_stride=args.v_stride
    )
