import torch
from lietorch import SE3, Sim3

MIN_DEPTH = 0.2


def extract_intrinsics(intrinsics):
    # breakpoint()
    return intrinsics[..., None, None, :].unbind(dim=-1)


def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float(),
    )

    return torch.stack([x, y], dim=-1)


def iproj(disps, intrinsics, jacobian=False):
    """pinhole camera inverse projection"""
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[..., -1] = 1.0
        return pts, J

    return pts, None


def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """pinhole camera projection"""
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5 * MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    coords = torch.stack([x, y, D * d], dim=-1) if return_depth else torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack(
            [
                fx * d,
                o,
                -fx * X * d * d,
                o,
                o,
                fy * d,
                -fy * Y * d * d,
                o,
                # o,     o,    -D*d*d,  d,
            ],
            dim=-1,
        ).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None


def actp(Gij, X0, jacobian=False):
    """action on point cloud"""
    X1 = Gij[:, :, None, None] * X0

    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    X,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    Y,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    Z,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None


def projective_transform(
    poses,
    depths,
    intrinsics,
    ii,
    jj,
    jacobian=False,
    return_depth=False,
    debug=False,
):
    """map points from ii->jj"""

    # inverse project (pinhole)
    X0, Jz = iproj(depths[:, ii], intrinsics[:, ii], jacobian=jacobian)

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    ## WHAT HACK IS THIS LINE?
    ## I think it's for stereo rig!
    # Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian)

    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:, jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    # if debug:
    # breakpoint()

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:, :, None, None, None].adjT(Jj)

        Jz = Gij[:, :, None, None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid


def iproj_f(disps, intrinsics, jacobian=False):
    """pinhole camera inverse projection"""
    ht, wd = disps.shape[2:]

    # print(intrinsics.shape)
    # breakpoint()
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        # assert fx == fy
        Jz = torch.zeros_like(pts)
        Jz[..., -1] = 1.0

        Jf = torch.zeros_like(pts[..., :3])
        Jf[..., 0] = -(x[None, None] - cx) / (fx) ** 2
        Jf[..., 1] = -(y[None, None] - cy) / (fy) ** 2

        return pts, Jz, Jf

    return pts, None, None


def focal_jacobian(intrinsics, P_j, Gij, disps, Jf):
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 0, 1]

    T_ij = Gij.matrix()
    d_XYZ = torch.matmul(T_ij[:, :, None, None, :3, :3], Jf.unsqueeze(-1)).squeeze(-1)
    d_XZ = (d_XYZ[..., 0] * P_j[..., 2] - P_j[..., 0] * d_XYZ[..., 2]) / P_j[..., 2] ** 2
    d_YZ = (d_XYZ[..., 1] * P_j[..., 2] - P_j[..., 1] * d_XYZ[..., 2]) / P_j[..., 2] ** 2

    return torch.stack(
        [
            P_j[..., 0] / P_j[..., 2] + fx * d_XZ,
            P_j[..., 1] / P_j[..., 2] + fy * d_YZ,
        ],
        dim=-1,
    ).unsqueeze(-1)


def projective_transform_f(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """map points from ii->jj"""

    # inverse project (pinhole)
    # breakpoint()
    X0, Jz, Jf = iproj_f(depths[:, ii], intrinsics[:, ii], jacobian=jacobian)

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    Gij.data[:, ii == jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian)

    # if np.abs(Gij[0, 1].matrix().cpu().numpy()[0, 3]) > 0.:
    # Tij = Gij[:, :, None, None].matrix()
    # X1_debug = Tij @ X0[..., None]
    # X1_debug2 = Tij[..., :3, :3] @ X0[..., :3, None] + X0[..., 3:4, None] * Tij[..., :3, 3:4]
    # breakpoint()

    # if np.abs(Gij[0, 1].matrix().cpu().numpy()[0, 3]) > 0.:
    # breakpoint()

    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:, jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:, :, None, None, None].adjT(Jj)

        Jz = Gij[:, :, None, None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        # Jacobian w.r.t focal length
        # Jf_1 = focal_jacobian_td(intrinsics[:,ii], X0, X1, Gij, depths[:,ii])
        Jf_1 = focal_jacobian(intrinsics[:, ii], X1, Gij, depths[:, ii], Jf)
        Jf_2 = Jf_1 * (intrinsics[:, ii, 2:3, None, None, None] * 2.0)

        return x1, valid, (Ji, Jj, Jz, Jf_2)

    return x1, valid


def induced_flow(poses, disps, intrinsics, ii, jj):
    """optical flow induced by camera motion"""

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[..., :2] - coords0, valid
