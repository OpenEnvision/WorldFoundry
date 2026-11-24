from factor_graph import FactorGraph

# isort: split
import torch


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        self.mean_Hessian = None

    @torch.no_grad()
    def __call__(
        self,
        steps=12,
        opt_intr=False,
        use_mono=True,
        alpha=0.005,
        ret_mask=False,
        ret_hessian=False,
    ):
        """main update"""
        t = self.video.counter.value
        if not self.video.stereo and not torch.any(self.video.disps_sens):
            self.video.normalize()

        graph = FactorGraph(
            self.video,
            self.update_op,
            corr_impl="alt",
            max_factors=16 * t,
            upsample=self.upsample,
        )

        graph.add_proximity_factors(
            rad=self.backend_radius,
            nms=self.backend_nms,
            thresh=self.backend_thresh,
            beta=self.beta,
        )

        torch_ba = True
        motion_prob = graph.update_lowmem(
            use_mono=use_mono,
            pytorch_ba=torch_ba,
            steps=steps,
            opt_intr=opt_intr,
            alpha=alpha,
            ret_mask=ret_mask,
        )
        median_hessian = None
        if ret_hessian:
            median_hessian = graph.estimate_preconditor()

        graph.clear_edges()
        self.video.dirty[:t] = True
        return median_hessian, motion_prob
