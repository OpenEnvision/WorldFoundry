from collections import OrderedDict

import torch

from worldfoundry.base_models.three_dimensions.slam.droid_slam.depth_video import DepthVideo
from worldfoundry.base_models.three_dimensions.slam.droid_slam.droid_backend import DroidBackend
from worldfoundry.base_models.three_dimensions.slam.droid_slam.droid_frontend import DroidFrontend
from worldfoundry.base_models.three_dimensions.slam.droid_slam.droid_net import DroidNet
from worldfoundry.base_models.three_dimensions.slam.droid_slam.motion_filter import MotionFilter
from worldfoundry.base_models.three_dimensions.slam.droid_slam.trajectory_filler import PoseTrajectoryFiller


class Droid:
    def __init__(self, args):
        super().__init__()
        self.load_weights(args.weights)
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

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

    def load_weights(self, weights):
        """load trained model weights"""

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([(k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """main thread - update map"""

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """terminate the visualization process, return poses [t, q]"""

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()
