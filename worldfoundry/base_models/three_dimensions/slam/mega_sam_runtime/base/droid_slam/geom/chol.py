import torch


class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1, -2))

        return dH, dz


def block_solve(H, b, ep=0.1, lm=0.0001):
    """solve normal equations"""
    B, N, _, D, _ = H.shape
    I = torch.eye(D).to(H.device)  # noqa: E741 - Ambiguous upstream name kept as written.
    H = H + (ep + lm * H) * I

    H = H.permute(0, 1, 3, 2, 4)
    H = H.reshape(B, N * D, N * D)
    b = b.reshape(B, N * D, 1)

    x = CholeskySolver.apply(H, b)
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, ep=0.01, lm=0.0001, sless=False):
    """solve using shur complement"""

    B, P, M, D, HW = E.shape

    # breakpoint()

    H = H.permute(0, 1, 3, 2, 4).reshape(B, P * D, P * D)
    E = E.permute(0, 1, 3, 2, 4).reshape(B, P * D, M * HW)
    Q = (1.0 / C).view(B, M * HW, 1)

    # damping
    I = torch.eye(P * D).to(H.device)  # noqa: E741 - Ambiguous upstream name kept as written.
    H = H + (ep + lm * H) * I

    v = v.reshape(B, P * D, 1)
    w = w.reshape(B, M * HW, 1)

    Et = E.transpose(1, 2)
    S = H - torch.matmul(E, Q * Et)
    v = v - torch.matmul(E, Q * w)

    # print("Condition number ", torch.linalg.cond(S, p=2) / (P * M))
    dx = CholeskySolver.apply(S.double(), v.double())
    dx = dx.to(Et.dtype)
    # breakpoint()

    if sless:
        return dx.reshape(B, P, D)

    dz = Q * (w - Et @ dx)
    dx = dx.reshape(B, P, D).float()
    dz = dz.reshape(B, M, HW).float()

    return dx, dz


def schur_solve_f(H, E, C, v, w, H_fc, Hff, vf, Efd, ep=0.01, lm=0.0001, sless=False):
    """solve using shur complement"""
    B, P, M, D, HW = E.shape
    H = H.permute(0, 1, 3, 2, 4).reshape(B, P * D, P * D)

    E = E.permute(0, 1, 3, 2, 4).reshape(B, P * D, M * HW)
    # Efd_copy = Efd.clone()
    Efd = Efd.permute(0, 1, 3, 2).reshape(B, M * HW, 1).transpose(-2, -1)

    # Augment top-right Hessian with focal length entry
    E = torch.cat([E, Efd], dim=-2)

    Q = (1.0 / C).view(B, M * HW, 1)

    # damping
    I = torch.eye(P * D).to(H.device)  # noqa: E741 - Ambiguous upstream name kept as written.
    H = H + (ep + lm * H) * I
    Hff = Hff + (ep + lm * Hff)

    # Augment Hessian with focal length entry
    H_fc = H_fc.permute(0, 1, 3, 2).reshape(B, P * D, 1)
    H = torch.cat([H, H_fc], dim=-1)
    H = torch.cat([H, torch.cat([H_fc.transpose(-2, -1), Hff], dim=-1)], dim=-2)

    # augment RHS of normal equation with focal residual
    v = v.reshape(B, P * D, 1)  # camera residual
    v = torch.cat([v, vf], dim=1)  #
    w = w.reshape(B, M * HW, 1)  # dispairty residaul

    Et = E.transpose(1, 2)

    _S_A = torch.matmul(E, Q * Et)
    _S_b = torch.matmul(E, Q * w)

    S = H - torch.matmul(E, Q * Et)
    v = v - torch.matmul(E, Q * w)

    dx = CholeskySolver.apply(S.double(), v.double()).to(Et.dtype)

    if sless:
        return dx.reshape(B, P, D)

    dz = Q * (w - Et @ dx.to(Et.dtype))

    dc = dx[:, :-1, :].reshape(B, P, D).float()
    df = dx[:, -1, :].float()
    dz = dz.reshape(B, M, HW).float()

    return dc, dz, df
