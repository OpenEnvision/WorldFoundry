# isort: split
from droid_net import DroidNet  # noqa: F401 - Keeps the module's import-time setup.

# isort: split
# isort: split
from factor_graph import FactorGraph

# isort: split
import lietorch
import torch
from lietorch import SE3


class PoseTrajectoryFiller:
    """This class is used to fill in non-keyframe poses"""

    def __init__(self, net, video, device="cuda:0"):
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.count = 0
        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """features for correlation volume"""
        return self.fnet(image)

    def __fill(self, tstamps, images, intrinsics, depths=None, masks=None):
        """fill operator"""

        tt = torch.as_tensor(tstamps, device="cuda")
        images = torch.stack(images, 0)
        intrinsics = torch.stack(intrinsics, 0)

        inputs = images[:, :, [2, 1, 0]].to(self.device) / 255.0

        if depths is not None:
            depths = torch.stack(depths, 0)

        if masks is not None:
            masks = torch.stack(masks, 0)  # motion mask

        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(tstamps)

        ts = self.video.tstamp[:N]
        Ps = SE3(self.video.poses[:N])

        t0 = torch.as_tensor([ts[ts <= t].shape[0] - 1 for t in tstamps])
        t1 = torch.where(t0 < N - 1, t0 + 1, t0)

        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(-1)
        w = v * (tt - ts[t0]).unsqueeze(-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        self.video.counter.value += M

        # breakpoint()

        self.video[N : N + M] = (
            tt,
            images[:, 0],
            Gs.data,
            1,
            depths,
            intrinsics / 8.0,
            masks,
            fmap,
        )

        graph = FactorGraph(self.video, self.update)
        graph.add_factors(t0.cuda(), torch.arange(N, N + M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N + M).cuda())

        for _itr in range(6):
            graph.update(N, N + M, motion_only=True, use_mono=True)

        Gs = SE3(self.video.poses[N : N + M].clone())
        self.video.counter.value -= M

        return [Gs]

    def __fill_full(self, tstamps, images, intrinsics, Gs, depths=None, masks=None):
        """fill operator"""

        tt = torch.as_tensor(tstamps, device="cuda")
        images = torch.stack(images, 0)
        if depths is not None:
            depths = torch.stack(depths, 0)  # sensor depths

        if masks is not None:
            masks = torch.stack(masks, 0)  # motion mask

        intrinsics = torch.stack(intrinsics, 0)
        inputs = images[:, :, [2, 1, 0]].to(self.device) / 255.0

        ### linear pose interpolation ###
        self.video.counter.value = len(tstamps)

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        self.video[0 : len(tstamps)] = (
            tt,
            images[:, 0],
            Gs.data,
            1,
            depths,
            intrinsics / 8.0,
            masks,
            fmap,
        )

    @torch.no_grad()
    def __call__(self, image_stream):
        """fill in poses of non-keyframe images"""

        # store all camera poses
        pose_list = []

        tstamps = []
        images = []
        depths = []
        masks = []
        intrinsics = []

        full_tstamps = []
        full_images = []
        full_depths = []
        full_masks = []
        full_intrinsics = []

        # self.video_full = copy.deepcopy(self.video)
        # First perform PGO over non-key frames
        # but keep track of all its entries so that we can do full BA later
        for tstamp, image, depth, _, mask in image_stream:
            intrinsic = self.video.intrinsics[0] * 8.0  # HARCODE!!

            tstamps.append(tstamp)
            images.append(image)
            depths.append(depth)
            masks.append(mask)
            intrinsics.append(intrinsic)

            # do PGO every 16 time steps
            if len(tstamps) == 16:
                pose_list += self.__fill(tstamps, images, intrinsics, depths=depths, masks=masks)
                full_tstamps += tstamps
                full_images += images
                full_depths += depths
                full_masks += masks
                full_intrinsics += intrinsics
                tstamps, images, depths, intrinsics = [], [], [], []
                masks = []

        if len(tstamps) > 0:
            full_tstamps += tstamps
            full_images += images
            full_depths += depths
            full_masks += masks
            full_intrinsics += intrinsics
            pose_list += self.__fill(tstamps, images, intrinsics, depths=depths, masks=masks)

        self.__fill_full(
            full_tstamps,
            full_images,
            full_intrinsics,
            lietorch.cat(pose_list, 0),
            depths=full_depths,
            masks=full_masks,
        )

        # stitch pose segments together
        return lietorch.cat(pose_list, 0)
