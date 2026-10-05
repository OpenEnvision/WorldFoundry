from collections import OrderedDict
from pathlib import Path

from depth_video import DepthVideo
from droid_backend import DroidBackend
from droid_frontend import DroidFrontend
from droid_net import DroidNet
import imageio
import lietorch
from motion_filter import MotionFilter
import numpy as np
import torch
from torch.multiprocessing import Process
from trajectory_filler import PoseTrajectoryFiller


class Droid:

  def __init__(self, args, net=None):
    super(Droid, self).__init__()
    if net is None:
      self.load_weights(args.weights)
    else:
      self.net = net
    self.args = args
    self.disable_vis = args.disable_vis

    # store images, depth, poses, intrinsics (shared between processes)
    self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

    # filter incoming frames so that there is enough motion
    self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

    # frontend process
    self.frontend = DroidFrontend(self.net, self.video, self.args)

    # backend process
    self.backend = DroidBackend(self.net, self.video, self.args)

    # visualizer
    # if not self.disable_vis:
    # from visualization import droid_visualization
    # self.visualizer = Process(target=droid_visualization, args=(self.video,))
    # self.visualizer.start()

    # post processor - fill in poses for non-keyframes
    self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

  def load_weights(self, weights):
    """load trained model weights"""

    print(weights)
    # print("multi_view_u ", multi_view_u)
    self.net = DroidNet()
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()
    ])

    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][
        :2
    ]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][
        :2
    ]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

    self.net.load_state_dict(state_dict, strict=True)
    self.net.to("cuda:0").eval()

  def track(self, tstamp, image, depth=None, intrinsics=None, mask=None):
    """main thread - update map"""

    with torch.no_grad():
      # check there is enough motion
      self.filterx.track(tstamp, image, depth, intrinsics, mask)
      # local bundle adjustment
      self.frontend()

  def track_final(self, tstamp, image, depth=None, intrinsics=None, mask=None):
    """main thread - update map"""
    # breakpoint()
    with torch.no_grad():
      # check there is enough motion
      self.filterx.track(
          tstamp, image, depth, intrinsics, mask, last_frame=True
      )
      # local bundle adjustment
      self.frontend(final_=True)

  def terminate(
      self,
      stream=None,
      _opt_intr=False,
      full_ba=False,
      scene_name=None,
      benchmark=False,
  ):
    """terminate the visualization process, return poses [t, q]"""
    del self.frontend
    torch.cuda.empty_cache()
    print("#" * 32)

    median_hessian = median_calib = 1e8
    prev_poses = self.video.poses.clone()
    prev_disps = self.video.disps.clone()
    prev_intrinsics = self.video.intrinsics.clone()
    alpha_base = 1e-5

    if not benchmark:
      median_stats, _ = self.backend(
          10, opt_intr=False, use_mono=True, alpha=alpha_base, ret_hessian=True
      )
      median_hessian, median_calib = median_stats
      print("median_hessian, median_calib ", median_hessian, median_calib)

    # we cannot observe focal length parameters
    if not benchmark and median_calib < 50:
      _opt_intr = False

    # we are not able to observe camera poses
    if not benchmark and median_hessian < 25:
      use_mono = True
      alpha = np.float32(
          alpha_base * np.exp(-(median_hessian.cpu().numpy() * 0.1))
      )
    else:
      use_mono = False
      alpha = 0.0

    print("_opt_intr ", _opt_intr, " use_mono ", use_mono, " alpha ", alpha)
    print(
        "################## KEYFRAME BUNDLE ADJUSTMENT " + "#" * 32,
        " use_mono ",
        use_mono,
    )

    self.video.poses = prev_poses.clone()
    self.video.disps = prev_disps.clone()
    self.video.intrinsics = prev_intrinsics.clone()
    self.backend(
        20,
        opt_intr=_opt_intr,
        use_mono=use_mono,
        alpha=alpha,
        ret_hessian=False,
    )

    camera_trajectory = self.traj_filler(stream)

    if full_ba:
      print("\nGLOBAL FRAME BA " + "#" * 32, " use_mono ", use_mono)
      torch.cuda.empty_cache()
      _, mot_prob = self.backend(
          15,
          opt_intr=False,
          use_mono=use_mono,
          alpha=alpha,
          ret_hessian=False,
          ret_mask=True,
      )

      opt_disps = self.video.disps[: self.video.counter.value, ...]
      full_ba_trajectory = lietorch.SE3(
          self.video.poses[: self.video.counter.value]
      )
      est_disps = torch.nn.functional.interpolate(
          opt_disps[:, None, ...], scale_factor=(8, 8), mode="bilinear"
      ).detach()  # + 1e-8

      return (
          full_ba_trajectory.data.cpu().numpy(),
          1.0 / est_disps[:, 0, ...].cpu().numpy(),
          mot_prob.cpu().numpy(),
      )
    else:
      opt_disps = self.video.disps_sens[: self.video.counter.value]
      est_disps = torch.nn.functional.interpolate(
          opt_disps[:, None, ...], scale_factor=(8, 8), mode="bilinear"
      ).detach()  # + 1e-8

      return (
          camera_trajectory.data.cpu().numpy(),
          1.0 / est_disps[:, 0, ...].cpu().numpy(),
      )
