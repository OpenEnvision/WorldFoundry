import droid_backends

# isort: split
from droid_net import cvx_upsample

# isort: split
# isort: split
import geom.ba as ba_pt

# isort: split
import geom.projective_ops as pops
import lietorch
import torch
from torch.multiprocessing import Value


class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):  # noqa: B006 - Keeps the upstream default object and signature.
        # current keyframe count
        self.counter = Value("i", 0)
        self.ready = Value("i", 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        # self.reg_type = reg_type
        # print("self.reg_type", reg_type)
        # breakpoint()
        self.model_id = 2

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht // 8, wd // 8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht // 8, wd // 8, device="cuda", dtype=torch.float).share_memory_()
        # self.disps_sens_hd = torch.zeros(buffer, ht//2, wd//2, device="cuda", dtype=torch.float).share_memory_()

        self.motion_masks = torch.zeros(buffer, ht // 8, wd // 8, device="cuda", dtype=torch.float).share_memory_()
        self.motion_w = torch.ones(buffer, ht // 8, wd // 8, device="cuda", dtype=torch.float).share_memory_()

        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics_init = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1

        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            # depth = item[4][3::8,3::8]
            if len(item[4].shape) == 3:
                dd = item[4][None]
            elif len(item[4].shape) == 2:
                dd = item[4][None, None]
            else:
                raise Exception

            depth_8x = torch.nn.functional.interpolate(
                dd, (item[1].shape[-2] // 8, item[1].shape[-1] // 8), mode="area"
            ).squeeze()
            # depth_2x = torch.nn.functional.interpolate(dd,
            # (item[1].shape[-2] // 2, item[1].shape[-1] // 2), mode='area').squeeze()

            self.disps[index] = torch.where(depth_8x > 0, 1.0 / depth_8x, 1e-3)
            self.disps_sens[index] = torch.where(depth_8x > 0, 1.0 / depth_8x, 0.0)

        if item[5] is not None:
            self.intrinsics[index] = item[5]
            self.intrinsics_init[index] = item[5]

        if item[6] is not None:
            if len(item[6].shape) == 3:
                mask = item[6][None]
            elif len(item[6].shape) == 2:
                mask = item[6][None, None]
            else:
                raise Exception

            # imageio.imwrite("./debug/mask_%02d.png"%int(self.tstamp[index]), np.uint8( 255. * mask[0, 0].cpu().numpy()))
            mask = torch.nn.functional.interpolate(
                mask,
                (item[1].shape[-2] // 8, item[1].shape[-1] // 8),
                mode="bilinear",
            ).squeeze()  # > 0.9
            # if USE_GT_MASK:
            self.motion_masks[index] = mask.float()

        if len(item) > 7:
            self.fmaps[index] = item[7]

        if len(item) > 8:
            self.nets[index] = item[8]

        if len(item) > 9:
            self.inps[index] = item[9]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """index the depth video"""

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index],
            )

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """to device, long, {-1}"""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """upsample disparity"""

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """normalize depth and poses"""
        with self.get_lock():
            s = self.disps[: self.counter.value].mean()
            self.disps[: self.counter.value] /= s
            self.poses[: self.counter.value, :3] *= s
            self.dirty[: self.counter.value] = True

    def reproject(self, ii, jj):
        """project points from ii -> jj"""
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """frame distance metric"""

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))

        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:
            poses = self.poses[: self.counter.value].clone()

            d1 = droid_backends.frame_distance(poses, self.disps, self.intrinsics[0], ii, jj, beta, self.model_id)

            d2 = droid_backends.frame_distance(poses, self.disps, self.intrinsics[0], jj, ii, beta, self.model_id)

            d = 0.5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses,
                self.disps,
                self.intrinsics[0],
                ii,
                jj,
                beta,
                self.model_id,
            )

        if return_matrix:
            return d.reshape(N, N)

        return d

    def estimate_preconditor(self, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """dense bundle adjustment (DBA)"""
        # [t0, t1] window of bundle adjustment optimization
        if t1 is None:
            t1 = max(ii.max().item(), jj.max().item()) + 1

        # if use_mono:
        _disps_sens = self.disps_sens
        # else: # not using mono depeth
        #     disps_sens = self.disps_sens * 0.

        Gs = lietorch.SE3(self.poses[None])

        median_hessian = ba_pt.compute_preconditioner(
            # target[None].permute(0, 1, 3, 4, 2),
            # weight[None].permute(0, 1, 3, 4, 2),
            Gs,
            # self.disps[None].clone(),
            self.intrinsics[None].clone(),
            self.disps[None],
            ii,
            jj,
            t0,
            t1,
            itrs,
            lm,
            ep,
        )

        return median_hessian

    def ba(
        self,
        target,
        weight,
        eta,
        ii,
        jj,
        t0=1,
        t1=None,
        itrs=2,
        lm=1e-4,
        ep=0.1,
        use_mono=False,
        motion_only=False,
        pytorch_ba=False,
        # mean_Hessian=None,
        mot_prob=None,
        opt_intr=False,
        alpha=0.05,
    ):
        # print("self.tstamp  ",  self.tstamp[:5])

        """dense bundle adjustment (DBA)"""
        empty_tensor = torch.empty(1)
        # print("alpha ", use_mono, alpha)
        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            if use_mono:  # noqa: SIM108 - Branches carry per-case comments.
                disps_sens = self.disps_sens
            else:  # not using mono depeth
                disps_sens = self.disps_sens * 0.0

            Gs = lietorch.SE3(self.poses[None])

            if opt_intr:
                # updated_poses, update_disps, error = ba_pt.precond_BA(
                #     mot_prob,
                #     target[None].permute(0, 1, 3, 4, 2),
                #     weight[None].permute(0, 1, 3, 4, 2),
                #     eta,
                #     Gs,
                #     self.disps[None].clone(),
                #     self.intrinsics[None].clone(),
                #     disps_sens[None],
                #     ii, jj,
                #     t0, t1,
                #     itrs,
                #     lm, ep,
                #     reg_type=self.reg_type)

                # self.poses = updated_poses.data[0].share_memory_()
                # self.disps = update_disps[0].share_memory_()

                # updated_poses_f, update_disps_f, updated_intrinsics_f, _ = ba_pt.BA_f(
                #     target[None].permute(0, 1, 3, 4, 2),
                #     weight[None].permute(0, 1, 3, 4, 2),
                #     eta,
                #     Gs,
                #     self.disps[None].clone(),
                #     self.intrinsics[None].clone(),
                #     disps_sens[None],
                #     self.intrinsics_init[None].clone(),
                #     ii, jj,
                #     t0, t1,
                #     itrs*2,
                #     lm, ep)

                # self.intrinsics = updated_intrinsics_f[0, 0:1].repeat(self.intrinsics.shape[0], 1).share_memory_()
                # self.poses = updated_poses_f.data[0].share_memory_()
                # self.disps = update_disps_f[0].share_memory_()

                DEBUG = True
                if DEBUG:
                    for _ in range(int(itrs * 2)):
                        Gs = lietorch.SE3(self.poses[None])

                        ret_calib = ba_pt.getJacobian(
                            target[None].permute(0, 1, 3, 4, 2),
                            weight[None].permute(0, 1, 3, 4, 2),
                            eta,
                            Gs,
                            self.disps[None].clone(),
                            self.intrinsics[None].clone(),
                            disps_sens[None],
                            self.intrinsics_init[None].clone(),
                            ii,
                            jj,
                            t0,
                            t1,
                            1,  # itrs*2,
                            lm,
                            ep,
                        )

                        Calib, CalibPose, CalibDepth, q_vec = (
                            ret_calib[0],
                            ret_calib[1],
                            ret_calib[2],
                            ret_calib[3],
                        )
                        Hs, vs, Eii, Eij, Cii, wi = (
                            ret_calib[4],
                            ret_calib[5],
                            ret_calib[6],
                            ret_calib[7],
                            ret_calib[8],
                            ret_calib[9],
                        )

                        droid_backends.ba(
                            self.poses,
                            self.disps,
                            self.intrinsics[0],
                            disps_sens,
                            target,
                            weight,
                            eta,
                            ii,
                            jj,
                            Calib,
                            CalibPose,
                            CalibDepth,
                            q_vec,
                            Hs,
                            vs,
                            Eii,
                            Eij,
                            Cii,
                            wi,
                            t0,
                            t1,
                            1,  # itrs*2,
                            self.model_id,
                            lm,
                            ep,
                            motion_only,
                            True,
                            alpha,
                        )

                        self.disps = torch.where(self.disps > 10, torch.zeros_like(self.disps), self.disps)
                        self.disps = self.disps.clamp(min=1e-6)
                        self.intrinsics = self.intrinsics[0:1].repeat(self.intrinsics.shape[0], 1)  # .share_memory_()
                        # self.intrinsics = updated_intrinsics_f[0, 0:1].repeat(self.intrinsics.shape[0], 1).share_memory_()
                        # self.poses = updated_poses_f.data[0].share_memory_()
                        # self.disps = update_disps_f[0].share_memory_()
                        # print("init, Opt focal ", self.intrinsics_init[0, 0], self.intrinsics[0, 0])

            else:
                # print("eta ", torch.mean(eta), lm, ep)
                droid_backends.ba(
                    self.poses,
                    self.disps,
                    self.intrinsics[0],
                    disps_sens,
                    target,
                    weight,
                    eta,
                    ii,
                    jj,
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                    t0,
                    t1,
                    itrs,
                    self.model_id,
                    lm,
                    ep,
                    motion_only,
                    False,
                    alpha,
                )

            error = torch.zeros(1)

            self.disps.clamp_(min=0.001)
            return error
