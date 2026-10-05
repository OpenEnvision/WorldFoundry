import cv2
import geom.projective_ops as pops
import lietorch
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.scatter import scatter_sum

from .chol import block_solve, schur_solve, schur_solve_f


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
  v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
  return scatter_sum(A[:, v], ii[v] * m + jj[v], dim=1, dim_size=n * m)


def safe_scatter_add_vec(b, ii, n):
  v = (ii >= 0) & (ii < n)
  return scatter_sum(b[:, v], ii[v], dim=1, dim_size=n)


# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
  ii = ii.to(device=dz.device)
  return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])


# apply retraction operator to poses
def pose_retr(poses, dx, ii):
  ii = ii.to(device=dx.device)
  return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


def focal_retr(intrinsics, df):
  focal_ratio = intrinsics[0, 0, 1] / intrinsics[0, 0, 0]
  return torch.cat(
      [
          intrinsics[:, :, 0:1] + df,
          intrinsics[:, :, 1:2] + df * focal_ratio,
          intrinsics[:, :, 2:],
      ],
      dim=-1,
  )


# def compute_error(A, sA):
# normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1),dim=1))
# error = A - torch.bmm(sA, sA)
# error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
# return torch.mean(error)

# def sqrt_newton_schulz(A, numIters, dtype):
#   batchSize = A.shape[0]
#   dim = A.shape[1]
#   normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
#   Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
#   I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
#   Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
#   for i in range(numIters):
#     T = 0.5*(3.0*I - Z.bmm(Y))
#     Y = Y.bmm(T)
#     Z = T.bmm(Z)
#     sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
#     error = compute_error(A, sA)
#     return sA, error

import numpy as np


@torch.no_grad()
def compute_preconditioner(
    poses, intrinsics, disps_sens, ii, jj, t0, t1, iterations, lm, ep
):
  """Full Bundle Adjustment"""
  B, _, ht, wd = disps_sens.shape
  P = t1 - t0
  N = ii.shape[0]

  ### 1: commpute jacobians and residuals ###
  coords, valid, (Ji, Jj, Jz, Jf) = pops.projective_transform_f(
      poses, disps_sens, intrinsics, ii, jj, jacobian=True
  )

  D = Ji.shape[-1]

  # r = (target - coords).contiguous().view(B, N, -1, 1).double()
  w = 1.0  # .001 * (valid * weight).contiguous().view(B, N, -1, 1).double()

  ### 2: construct linear system ###
  Ji = Ji.reshape(B, N, -1, D).float()
  Jj = Jj.reshape(B, N, -1, D).float()
  wJiT = (w * Ji).transpose(2, 3)
  wJjT = (w * Jj).transpose(2, 3)

  Jz = Jz.reshape(B, N, ht * wd, -1)

  # Hii = torch.matmul(wJiT, Ji)
  # Hij = torch.matmul(wJiT, Jj)
  # Hji = Hij.transpose(-2, -1) #torch.matmul(wJjT, Ji)
  # Hjj = torch.matmul(wJjT, Jj)

  # Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
  # Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

  Ck = torch.sum(w * Jz * Jz, dim=-1)

  kx, kk = torch.unique(ii, return_inverse=True)
  M = kx.shape[0]

  ii_ = ii - t0  # line 1378 in cuda code
  jj_ = jj - t0

  C_0 = safe_scatter_add_vec(Ck, kk, M)
  # w = safe_scatter_add_vec(wk, kk, M)

  if False:
    C_0_viz = C_0.reshape(-1, C_0.shape[1], ht, wd)
    C_0_viz = C_0_viz[0, 1].cpu().numpy()
    C_0_viz = cv2.resize(
        C_0_viz, (C_0_viz.shape[-1] * 8, C_0_viz.shape[-2] * 8)
    )
    C_0_viz_inv = 1.0 / (C_0_viz + 1e-4)
    # C_0_viz_inv[C_0_viz_inv > 0.175] = 0.
    # Create the plot
    fig, ax = plt.subplots(
        figsize=(584 / 128, 328 / 128), dpi=128
    )  # Set figure size in inches
    # Plot the probability map
    im = ax.imshow(C_0_viz_inv, cmap='jet', vmin=0.0)
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    plt.axis('off')
    plt.tight_layout()
    # cbar.set_label('Probability')
    # Add labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('2D Probability Map')
    plt.savefig('debug/hessian_lab-coat.png')
    breakpoint()

  median_Hessian = torch.median(C_0)  # / (N)

  Jf = Jf.reshape(B, N, -1, 1).float()

  wJfT = (w * Jf).transpose(2, 3)  # .view(B, -1, 1).transpose(-2, -1)

  # information within intrinsics, correct
  Hff = torch.matmul(
      wJfT.transpose(-2, -1).view(B, -1, 1).transpose(-2, -1), Jf.view(B, -1, 1)
  )

  Calib = (Hff).float()
  median_Calib = torch.median(Calib) / (intrinsics[:, 0, 2] * 2.0) ** 2

  return [median_Hessian, median_Calib]


@torch.no_grad()
def precond_BA(
    mot_prob,  # P_inv_pose, P_inv_disp, mean_Hessian,
    target,
    weight,
    eta,
    poses,
    disps,
    intrinsics,
    disps_sens,
    ii,
    jj,
    t0,
    t1,
    iterations,
    lm,
    ep,
    reg_type,
):
  """Full Bundle Adjustment"""

  B, _, ht, wd = disps.shape
  P = t1 - t0
  N = ii.shape[0]

  # align estimated dispairty with mono-dispairty with precomputed scale
  disps = disps.clamp(min=1e-2, max=10.0)
  disps_sens = disps_sens.clamp(min=1e-2, max=10.0)

  kx, kk = torch.unique(ii, return_inverse=True)
  # disps[: kx]

  # if reg_type == 'log':
  #     log_diff = torch.log(disps_sens[:, kx]) - torch.log(disps[:, kx])
  #     align_scale , _ = torch.median(log_diff.view(kx.shape[0], -1), dim=-1)#.detach()
  #     align_scale = align_scale#.detach()
  # elif reg_type == 'linear':
  #     ratio = disps_sens[:, kx] / disps[:, kx]#.clamp(min=1e-3)
  #     align_scale, _  = torch.median(ratio.view(kx.shape[0], -1), dim=-1)
  #     align_scale = align_scale#.detach()

  # damping_scale = torch.exp(-mean_Hessian * 4. / 10.)
  damping_scale = 0.0  # torch.exp(-mean_Hessian * 4.)
  # print("mean_Hessian ", N, P, mean_Hessian, damping_scale)
  # precondition = False

  # if precondition:
  #     P_inv_pose_ii = torch.index_select(P_inv_pose, dim=1, index=ii)
  #     P_inv_pose_ii = P_inv_pose_ii[:, :, None, None, :, :].expand(1, -1, ht, wd, 6, 6)
  #     P_inv_pose_jj = torch.index_select(P_inv_pose, dim=1, index=jj)
  #     P_inv_pose_jj = P_inv_pose_jj[:, :, None, None, :, :].expand(1, -1, ht, wd, 6, 6)

  #     P_inv_disp_ii = torch.index_select(P_inv_disp, dim=1, index=ii)
  #     P_inv_disp_ii = P_inv_disp_ii.view(P_inv_disp_ii.shape[0], P_inv_disp_ii.shape[1], ht, wd)[..., None, None] #expand(1, ii.shape[0], ht, wd, 6, 6)

  for itr in range(iterations):
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True
    )

    # if precondition:
    #     # multiply with precondiong matrix
    #     Ji = Ji @ P_inv_pose_ii
    #     Jj = Jj @ P_inv_pose_jj
    #     Jz = Jz * P_inv_disp_ii

    D = Ji.shape[-1]

    r = (target - coords).contiguous().view(B, N, -1, 1).float()
    w = 0.001 * (valid * weight).contiguous().view(B, N, -1, 1).float()

    # breakpoint()
    # cost = torch.sum((valid * weight) * (target - coords)**2)/torch.sum(valid * weight)
    # print("non-focal residual ", cost)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D).float()
    Jj = Jj.reshape(B, N, -1, D).float()
    wJiT = (w * Ji).transpose(2, 3)
    wJjT = (w * Jj).transpose(2, 3)

    Jz = Jz.reshape(B, N, ht * wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = Hij.transpose(-2, -1)  # torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    Ej = (wJjT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)

    w = w.view(B, N, ht * wd, -1)
    r = r.view(B, N, ht * wd, -1)
    wk = torch.sum(w * r * Jz, dim=-1)
    Ck = torch.sum(w * Jz * Jz, dim=-1)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    # P = P // rig - fixedp
    # ii = ii // rig - fixedp
    # jj = jj // rig - fixedp

    ii_ = ii - t0  # line 1378 in cuda code
    jj_ = jj - t0

    # breakpoint()
    H = (
        safe_scatter_add_mat(Hii, ii_, ii_, P, P)
        + safe_scatter_add_mat(Hij, ii_, jj_, P, P)
        + safe_scatter_add_mat(Hji, jj_, ii_, P, P)
        + safe_scatter_add_mat(Hjj, jj_, jj_, P, P)
    )

    E = safe_scatter_add_mat(Ei, ii_, kk, P, M) + safe_scatter_add_mat(
        Ej, jj_, kk, P, M
    )
    # breakpoint()

    v = safe_scatter_add_vec(vi, ii_, P) + safe_scatter_add_vec(vj, jj_, P)

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    # print("non/focal E", E[0, 2, 3, 20:30])
    # print("non/focal C", C[0, 2, 30:40])
    # breakpoint()

    # // add depth residual if there are depth sensor measurements
    # torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
    # torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
    # torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
    # alpha = 0.05 * damping_scale
    # m = (disps_sens[:, kx, ...] > 1e-8).float().view(*C_0.shape)

    # breakpoint()
    # why we can do direct addition? Because we only save diagonal non-zero entry!
    # C = C_0 + m * alpha + (1 - alpha) * (eta.view(*C_0.shape)) + 1e-7
    # This is L method
    # C = C_0 + m * alpha + (1 - m) * (eta.view(*C_0.shape)) + (1 - m) * alpha * damping_scale + 1e-7
    # Thisa is M method
    # C = C_0 + m * alpha + (1 - m) * (eta.view(*C_0.shape)) + (1 - m) * C_0 / mean_Hessian + 1e-7

    # w = w - m * alpha * (disps[:, kx] - disps_sens[:, kx]).view(*w.shape)
    # why we can do direct addition? Because we only save diagonal non-zero entry!
    # This is for disparity L2 loss

    # if reg_type == 'log':
    #     # This is for log disaprity L2 loss, which is perfectly scale invariance!!!
    #     alpha = 0.01
    #     m = mot_prob.float().view(*C.shape) #(disps_sens[:, kx, ...] > 1e-8).float().view(*C.shape)

    #     J_reg = (1./ disps[:, kx]).view(*C.shape)
    #     C = C + m * alpha * J_reg + (1 - m) * (eta.view(*C.shape)) + 1e-7
    #     w = w - m * alpha * J_reg * (torch.log(disps[:, kx]) + align_scale[None, :, None, None] - torch.log(disps_sens[:, kx])).view(*w.shape)
    #     # print(torch.mean(torch.abs((torch.log(disps[:, kx]) + align_scale[None, :, None, None] - torch.log(disps_sens[:, kx])).view(*w.shape))))
    # elif reg_type == 'linear':
    #     # This is for disparity L2 loss, which is scale aligned
    #     alpha = 0.05
    #     m = (disps_sens[:, kx, ...] > 1e-8).float().view(*C.shape)
    #     J_reg = 1.
    #     C = C + m * alpha * J_reg + (1 - m) * (eta.view(*C.shape)) + 1e-7
    #     w = w - m * alpha * J_reg * (disps[:, kx] * align_scale[None, :, None, None] - disps_sens[:, kx]).view(*w.shape)
    # else:
    #     # This is Vanila version!!
    #     C = C + eta.view(*C.shape) + 1e-7

    # This is for disparity L2 loss, which is scale aligned
    alpha = 0.05
    m = (disps_sens[:, kx, ...] > 1e-8).float().view(*C.shape)
    C = C + m * alpha + (1 - m) * (eta.view(*C.shape)) + 1e-7
    w = w - m * alpha * (disps[:, kx] - disps_sens[:, kx]).view(*w.shape)

    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht * wd)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w, ep=ep, lm=lm)

    # print("pytorch dx ", dx[0, 0], dx.shape)
    # print("pytorch dx ", dx[0, 1], dx.shape)

    # breakpoint()

    ### 4: apply retraction ###
    # WE NEED CONVERT BACK TO ORIGINAL SPACE!
    # if precondition:
    # dx = P_inv_pose[:, t0:, ...] @ dx[..., None].float()
    # dx = dx.squeeze(-1)
    # dz = P_inv_disp * dz.float()

    poses = pose_retr(poses, dx.float(), torch.arange(P) + t0)
    disps = disp_retr(disps, dz.float().view(B, -1, ht, wd), kx)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=1e-6)

  cost = torch.sum((valid * weight) * (target - coords) ** 2) / torch.sum(
      valid * weight
  )
  # print("non-focal residual ", cost)

  return poses, disps, cost


@torch.no_grad()
def BA_f(
    target,
    weight,
    eta,
    poses,
    disps,
    intrinsics,
    disps_sens,
    intrinsics_init,
    ii,
    jj,
    t0,
    t1,
    iterations,
    lm,
    ep,
):
  """Full Bundle Adjustment with focal length optimization

  Be very carefull how Hessian is constructed from Jacobian
  """
  B, _, ht, wd = disps.shape
  P = t1 - t0
  N = ii.shape[0]

  for itr in range(iterations):
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz, Jf) = pops.projective_transform_f(
        poses, disps, intrinsics, ii, jj, jacobian=True
    )

    # print("Jf ", Jf, Jf.shape)
    D = Ji.shape[-1]

    r = (target - coords).contiguous().view(B, N, -1, 1).float()
    w = 0.001 * (valid * weight).contiguous().view(B, N, -1, 1).float()

    cost = torch.sum((valid * weight) * (target - coords) ** 2) / torch.sum(
        valid * weight
    )
    # print("cost focal ", torch.mean(cost))

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D).float()
    Jj = Jj.reshape(B, N, -1, D).float()
    Jf = Jf.reshape(B, N, -1, 1).float()

    wJiT = (w * Ji).transpose(2, 3)
    wJjT = (w * Jj).transpose(2, 3)
    wJfT = (w * Jf).transpose(2, 3)  # .view(B, -1, 1).transpose(-2, -1)
    # wJiT = (Ji).transpose(2,3)
    # wJjT = (Jj).transpose(2,3)
    # wJfT = (Jf).transpose(2,3) #.view(B, -1, 1).transpose(-2, -1)

    Jz = Jz.reshape(B, N, ht * wd, -1).float()

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = Hij.transpose(-2, -1)  # torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # information matrix between intrinsic and poses
    Hfi = torch.matmul(wJfT, Ji)
    Hfj = torch.matmul(wJfT, Jj)

    CalibPose = (
        torch.cat([Hfi, Hfj], dim=2).permute(1, 2, 0, 3).float()
    )  # {num, 2, n_intr, 6}

    # information within intrinsics, correct
    Hff = torch.matmul(
        wJfT.transpose(-2, -1).view(B, -1, 1).transpose(-2, -1),
        Jf.view(B, -1, 1),
    )

    Calib = (Hff + 0.0).float()

    # print("Hff ", Hff, Hff.shape)
    # breakpoint()
    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)
    # RHS: J^T * W * r
    vf = torch.matmul(
        wJfT.transpose(-2, -1).view(B, -1, 1).transpose(-2, -1),
        r.view(B, -1, 1),
    )  # .squeeze(-1)

    alpha_f = 0.0
    Hff = Hff + alpha_f
    vf = vf - alpha_f * (intrinsics[0, 0, 0] - intrinsics_init[0, 0, 0])

    q_vec = vf[..., -1].float()
    # print("vf ", torch.sum(vf), vf.shape)

    Ei = (wJiT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    Ej = (wJjT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    # information matrix betwen intrinsics and dispairty
    Ef = (wJfT.view(B, N, 1, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    CalibDepth = Ef.permute(1, 3, 0, 2)[..., -1].float()  # num, ht*wd, n_intr
    # print("Ef ", torch.sum(Ef), Ef.shape)
    # breakpoint()

    w = w.view(B, N, ht * wd, -1)
    r = r.view(B, N, ht * wd, -1)
    wk = torch.sum(w * r * Jz, dim=-1)
    Ck = torch.sum(w * Jz * Jz, dim=-1)

    # only optimize keyframe poses
    # P = P // rig - fixedp
    # ii = ii // rig - fixedp
    # jj = jj // rig - fixedp

    ii_ = ii - t0  # line 1378 in cuda code
    jj_ = jj - t0

    # Check correct!
    H_fc = safe_scatter_add_vec(Hfi, ii_, P) + safe_scatter_add_vec(Hfj, jj_, P)

    # print("H_fc", H_fc.view(-1, 1), H_fc.shape)

    H = (
        safe_scatter_add_mat(Hii, ii_, ii_, P, P)
        + safe_scatter_add_mat(Hij, ii_, jj_, P, P)
        + safe_scatter_add_mat(Hji, jj_, ii_, P, P)
        + safe_scatter_add_mat(Hjj, jj_, jj_, P, P)
    )

    Hs = torch.cat([Hii, Hij, Hji, Hjj], dim=0).float()

    # torch::Tensor Hs = torch::zeros({4, num, 6, 6}, opts);
    # torch::Tensor vs = torch::zeros({2, num, 6}, opts);
    # torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts);
    # torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts);

    # print("Hs ", torch.sum(Hii) + torch.sum(Hij) + torch.sum(Hji) + torch.sum(Hjj))

    Efd = safe_scatter_add_vec(Ef, kk, M)

    E = safe_scatter_add_mat(Ei, ii_, kk, P, M) + safe_scatter_add_mat(
        Ej, jj_, kk, P, M
    )

    Eii = Ei[0].float()
    Eij = Ej[0].float()

    # breakpoint()
    # print("Ei ", torch.sum(Ei), " Ej ", torch.sum(Ej))

    v = safe_scatter_add_vec(vi, ii_, P) + safe_scatter_add_vec(vj, jj_, P)

    # breakpoint()
    vs = torch.cat([vi, vj], dim=0).float()

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    Cii = Ck[0].float()
    wi = wk[0].float()
    # breakpoint()
    # torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
    # torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

    # print("C ", torch.sum(C), C.shape)
    # print("w ", torch.sum(w), w.shape)

    # print("focal E", E[0, 2, 3, 20:30])
    # print("focal C", C[0, 2, 30:40])
    # breakpoint()

    # // add depth residual if there are depth sensor measurements
    # torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
    # torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
    # torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
    alpha = 0.05
    m = (disps_sens[:, kx, ...] > 1e-8).float().view(*C.shape)
    # # why we can do direct addition? Because we only save diagonal non-zero entry!
    C = C + m * alpha + (1 - m) * eta.view(*C.shape) + 1e-7
    w = w - m * alpha * (disps[:, kx] - disps_sens[:, kx]).view(*w.shape)
    # C = C + eta.view(*C.shape) + 1e-7

    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht * wd)

    ### 3: solve the system ###
    dx, dz, df = schur_solve_f(H, E, C, v, w, H_fc, Hff, vf, Efd, ep=ep, lm=lm)
    # dx_, dz_ = schur_solve(H, E, C, v, w,
    # ep=ep,
    # lm=lm)
    # print("df ", df)
    ### 4: apply retraction ###
    poses = pose_retr(poses, dx.float(), torch.arange(P) + t0)
    disps = disp_retr(disps, dz.float().view(B, -1, ht, wd), kx)

    # intrinsics = focal_retr(intrinsics, df)
    # print("df ", df, df * (intrinsics[0, 0, 2] * 2))
    intrinsics = focal_retr(intrinsics, df * (intrinsics[0, 0, 2] * 2))

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=1e-6)

  return (
      poses,
      disps,
      intrinsics,
      [Calib, CalibPose, CalibDepth, q_vec, Hs, vs, Eii, Eij, Cii, wi],
  )  # , torch.mean(cost)


@torch.no_grad()
def getJacobian(
    target,
    weight,
    eta,
    poses,
    disps,
    intrinsics,
    disps_sens,
    intrinsics_init,
    ii,
    jj,
    t0,
    t1,
    iterations,
    lm,
    ep,
):
  """Compute Focal included Jacobian for second order BA"""
  B, _, ht, wd = disps.shape
  P = t1 - t0
  N = ii.shape[0]

  for itr in range(iterations):
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz, Jf) = pops.projective_transform_f(
        poses, disps, intrinsics, ii, jj, jacobian=True
    )

    # print("Jf ", Jf, Jf.shape)
    D = Ji.shape[-1]

    r = (target - coords).contiguous().view(B, N, -1, 1).float()
    w = 0.001 * (valid * weight).contiguous().view(B, N, -1, 1).float()

    # cost = torch.sum((valid * weight) * (target - coords)**2) / torch.sum(valid * weight)
    # print("cost focal ", torch.mean(cost))

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D).float()
    Jj = Jj.reshape(B, N, -1, D).float()
    Jf = Jf.reshape(B, N, -1, 1).float()

    wJiT = (w * Ji).transpose(2, 3)
    wJjT = (w * Jj).transpose(2, 3)
    wJfT = (w * Jf).transpose(2, 3)  # .view(B, -1, 1).transpose(-2, -1)
    # wJiT = (Ji).transpose(2,3)
    # wJjT = (Jj).transpose(2,3)
    # wJfT = (Jf).transpose(2,3) #.view(B, -1, 1).transpose(-2, -1)

    Jz = Jz.reshape(B, N, ht * wd, -1).float()

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = Hij.transpose(-2, -1)  # torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)
    Hs = torch.cat([Hii, Hij, Hji, Hjj], dim=0).float()

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # information matrix between intrinsic and poses
    Hfi = torch.matmul(wJfT, Ji)
    Hfj = torch.matmul(wJfT, Jj)
    CalibPose = (
        torch.cat([Hfi, Hfj], dim=2).permute(1, 2, 0, 3).float()
    )  # {num, 2, n_intr, 6}

    # information within intrinsics, correct
    Hff = torch.matmul(
        wJfT.transpose(-2, -1).view(B, -1, 1).transpose(-2, -1),
        Jf.view(B, -1, 1),
    )

    Calib = (Hff + 0.0).float()

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)
    # RHS: J^T * W * r
    vf = torch.matmul(
        wJfT.transpose(-2, -1).view(B, -1, 1).transpose(-2, -1),
        r.view(B, -1, 1),
    )  # .squeeze(-1)

    alpha_f = 0.0
    Hff = Hff + alpha_f
    vf = vf - alpha_f * (intrinsics[0, 0, 0] - intrinsics_init[0, 0, 0])

    q_vec = vf[..., -1].float()

    Ei = (wJiT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    Ej = (wJjT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    # information matrix betwen intrinsics and dispairty
    Ef = (wJfT.view(B, N, 1, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    CalibDepth = Ef.permute(1, 3, 0, 2)[..., -1].float()  # num, ht*wd, n_intr

    w = w.view(B, N, ht * wd, -1)
    r = r.view(B, N, ht * wd, -1)
    wk = torch.sum(w * r * Jz, dim=-1)
    Ck = torch.sum(w * Jz * Jz, dim=-1)

    # only optimize keyframe poses
    # P = P // rig - fixedp
    # ii = ii // rig - fixedp
    # jj = jj // rig - fixedp

    ii_ = ii - t0  # line 1378 in cuda code
    jj_ = jj - t0

    # Check correct!
    # H_fc = safe_scatter_add_vec(Hfi, ii_, P) + \
    # safe_scatter_add_vec(Hfj, jj_, P)

    # H = safe_scatter_add_mat(Hii, ii_, ii_, P, P) + \
    # safe_scatter_add_mat(Hij, ii_, jj_, P, P) + \
    # safe_scatter_add_mat(Hji, jj_, ii_, P, P) + \
    # safe_scatter_add_mat(Hjj, jj_, jj_, P, P)

    # Efd = safe_scatter_add_vec(Ef, kk, M)

    # E = safe_scatter_add_mat(Ei, ii_, kk, P, M) + \
    # safe_scatter_add_mat(Ej, jj_, kk, P, M)

    Eii = Ei[0].float()
    Eij = Ej[0].float()

    # breakpoint()
    # print("Ei ", torch.sum(Ei), " Ej ", torch.sum(Ej))

    # v = safe_scatter_add_vec(vi, ii_, P) + \
    # safe_scatter_add_vec(vj, jj_, P)

    # breakpoint()
    vs = torch.cat([vi, vj], dim=0).float()

    # C = safe_scatter_add_vec(Ck, kk, M)
    # w = safe_scatter_add_vec(wk, kk, M)

    Cii = Ck[0].float()
    wi = wk[0].float()

  return [
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
  ]  # , torch.mean(cost)


@torch.no_grad()
def BA(
    target,
    weight,
    mot_prob,
    eta,
    poses,
    disps,
    intrinsics,
    disps_sens,
    ii,
    jj,
    t0,
    t1,
    iterations,
    lm,
    ep,
    reg_type=None,
):
  """Full Bundle Adjustment"""

  B, _, ht, wd = disps.shape
  P = t1 - t0
  N = ii.shape[0]

  for itr in range(iterations):
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True
    )

    D = Ji.shape[-1]

    r = (target - coords).contiguous().view(B, N, -1, 1).double()
    w = 0.001 * (valid * weight).contiguous().view(B, N, -1, 1).double()

    # breakpoint()
    cost = (0.001 * (valid * weight)) * (target - coords) ** 2
    print('residual ', torch.mean(cost))

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D).double()
    Jj = Jj.reshape(B, N, -1, D).double()
    wJiT = (w * Ji).transpose(2, 3)
    wJjT = (w * Jj).transpose(2, 3)
    # wJiT = (Ji).transpose(2,3)
    # wJjT = (Jj).transpose(2,3)

    # J_debug = torch.zeros((Jj.shape[1] * Jj.shape[2], P * D)).double().cuda()
    # ii_ = ii - t0
    # jj_ = jj - t0
    # for k in range(Ji.shape[1]):
    #     for l in range(Ji.shape[2]):
    #         print(k, l)
    #         if 0 <= ii_[k] < P and 0 <= jj_[k] < P:
    #             J_debug[k * Ji.shape[2] + l, ii_[k]*D:(ii_[k]+1)*D] += Ji[0, k, l, :].double()
    #             J_debug[k * Ji.shape[2] + l, jj_[k]*D:(jj_[k]+1)*D] += Jj[0, k, l, :].double()
    #             continue
    #         if 0 <= ii_[k] < P:
    #             J_debug[k * Ji.shape[2] + l, ii_[k]*D:(ii_[k]+1)*D] += Ji[0, k, l, :].double()
    #             continue
    #         if 0 <= jj_[k] < P:
    #             J_debug[k * Ji.shape[2] + l, jj_[k]*D:(jj_[k]+1)*D] += Jj[0, k, l, :].double()
    #             continue

    # H_debug = J_debug.transpose(0, 1) @ J_debug

    Jz = Jz.reshape(B, N, ht * wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = Hij.transpose(-2, -1)  # torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    Ej = (wJjT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)

    w = w.view(B, N, ht * wd, -1)
    r = r.view(B, N, ht * wd, -1)
    wk = torch.sum(w * r * Jz, dim=-1)
    Ck = torch.sum(w * Jz * Jz, dim=-1)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    # P = P // rig - fixedp
    # ii = ii // rig - fixedp
    # jj = jj // rig - fixedp

    ii_ = ii - t0  # line 1378 in cuda code
    jj_ = jj - t0

    # breakpoint()

    H = (
        safe_scatter_add_mat(Hii, ii_, ii_, P, P)
        + safe_scatter_add_mat(Hij, ii_, jj_, P, P)
        + safe_scatter_add_mat(Hji, jj_, ii_, P, P)
        + safe_scatter_add_mat(Hjj, jj_, jj_, P, P)
    )

    E = safe_scatter_add_mat(Ei, ii_, kk, P, M) + safe_scatter_add_mat(
        Ej, jj_, kk, P, M
    )
    # breakpoint()

    v = safe_scatter_add_vec(vi, ii_, P) + safe_scatter_add_vec(vj, jj_, P)
    # breakpoint()

    C_0 = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    # // add depth residual if there are depth sensor measurements
    # torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
    # torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
    # torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
    alpha = 0.05
    m = (disps_sens[:, kx, ...] > 1e-8).float().view(*C_0.shape)

    # why we can do direct addition? Because we only save diagonal non-zero entry!
    C = C_0 + m * alpha + (1 - m) * (eta.view(*C_0.shape)) + 1e-7
    w = w - m * alpha * (disps[:, kx] - disps_sens[:, kx]).view(*w.shape)

    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht * wd)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w, ep=ep, lm=lm)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx.float(), torch.arange(P) + t0)
    disps = disp_retr(disps, dz.float().view(B, -1, ht, wd), kx)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=1e-6)

  return poses, disps


def BA_f_train(
    target,
    weight,
    eta,
    poses,
    disps,
    disps_sens,
    intrinsics,
    ii,
    jj,  # , t0, t1,
    fixedp=1,
    rig=1,
):
  """Full Bundle Adjustment with focal length optimization

  Be very carefull how Hessian is constructed from Jacobian
  """

  B, P, ht, wd = disps.shape
  # B, _, ht, wd = disps.shape
  # P = t1 - t0
  N = ii.shape[0]
  # D = poses.manifold_dim

  ### 1: commpute jacobians and residuals ###
  coords, valid, (Ji, Jj, Jz, Jf) = pops.projective_transform_f(
      poses, disps, intrinsics, ii, jj, jacobian=True
  )

  D = Ji.shape[-1]

  r = (target - coords).contiguous().view(B, N, -1, 1)  # .double()
  w = 0.001 * (valid * weight).contiguous().view(B, N, -1, 1)  # .double()

  # cost = torch.sum((valid * weight) * (target - coords)**2)/torch.sum(valid * weight)
  # print("w/ focal residual ", cost)

  ### 2: construct linear system ###
  Ji = Ji.reshape(B, N, -1, D)  # .double()
  Jj = Jj.reshape(B, N, -1, D)  # .double()
  Jf = Jf.reshape(B, N, -1, 1)  # .double()

  wJiT = (w * Ji).transpose(2, 3)
  wJjT = (w * Jj).transpose(2, 3)
  wJfT = (w * Jf).transpose(2, 3)  # .view(B, -1, 1).transpose(-2, -1)
  # wJiT = (Ji).transpose(2,3)
  # wJjT = (Jj).transpose(2,3)
  # wJfT = (Jf).transpose(2,3) #.view(B, -1, 1).transpose(-2, -1)

  Jz = Jz.reshape(B, N, ht * wd, -1)  # .double()

  Hii = torch.matmul(wJiT, Ji)
  Hij = torch.matmul(wJiT, Jj)
  Hji = Hij.transpose(-2, -1)  # torch.matmul(wJjT, Ji)
  Hjj = torch.matmul(wJjT, Jj)

  kx, kk = torch.unique(ii, return_inverse=True)
  M = kx.shape[0]

  # information matrix between intrinsic and poses
  Hfi = torch.matmul(wJfT, Ji)
  Hfj = torch.matmul(wJfT, Jj)
  # information within intrinsics, correct
  Hff = torch.matmul(
      wJfT.transpose(-2, -1).view(B, -1, 1).transpose(-2, -1), Jf.view(B, -1, 1)
  )

  vi = torch.matmul(wJiT, r).squeeze(-1)
  vj = torch.matmul(wJjT, r).squeeze(-1)
  # RHS: J^T * W * r
  vf = torch.matmul(
      wJfT.transpose(-2, -1).view(B, -1, 1).transpose(-2, -1), r.view(B, -1, 1)
  )  # .squeeze(-1)

  # alpha_f = 0.1
  # Hff = Hff + alpha_f
  # vf = vf - alpha_f * (intrinsics[0, 0, 0] - intrinsics_init[0, 0, 0])#.view(*w.shape)

  Ei = (wJiT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
  Ej = (wJjT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
  # information matrix betwen intrinsics and dispairty
  Ef = (wJfT.view(B, N, 1, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)

  w = w.view(B, N, ht * wd, -1)
  r = r.view(B, N, ht * wd, -1)
  wk = torch.sum(w * r * Jz, dim=-1)
  Ck = torch.sum(w * Jz * Jz, dim=-1)

  # only optimize keyframe poses
  P = P // rig - fixedp
  ii_ = ii // rig - fixedp
  jj_ = jj // rig - fixedp

  # ii_ = ii - t0 # line 1378 in cuda code
  # jj_ = jj - t0
  # Check correct!
  H_fc = safe_scatter_add_vec(Hfi, ii_, P) + safe_scatter_add_vec(Hfj, jj_, P)

  H = (
      safe_scatter_add_mat(Hii, ii_, ii_, P, P)
      + safe_scatter_add_mat(Hij, ii_, jj_, P, P)
      + safe_scatter_add_mat(Hji, jj_, ii_, P, P)
      + safe_scatter_add_mat(Hjj, jj_, jj_, P, P)
  )

  Efd = safe_scatter_add_vec(Ef, kk, M)

  E = safe_scatter_add_mat(Ei, ii_, kk, P, M) + safe_scatter_add_mat(
      Ej, jj_, kk, P, M
  )
  # breakpoint()

  v = safe_scatter_add_vec(vi, ii_, P) + safe_scatter_add_vec(vj, jj_, P)
  # breakpoint()

  C = safe_scatter_add_vec(Ck, kk, M)
  w = safe_scatter_add_vec(wk, kk, M)

  # // add depth residual if there are depth sensor measurements
  # torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
  # torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
  # torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
  # alpha = 0.05
  # m = (disps_sens[:, kx, ...] > 1e-8).float().view(*C.shape)
  # why we can do direct addition? Because we only save diagonal non-zero entry!
  C = C + eta.view(*C.shape) + 1e-7
  # w = w

  H = H.view(B, P, P, D, D)
  E = E.view(B, P, M, D, ht * wd)

  ### 3: solve the system ###
  dx, dz, df = schur_solve_f(H, E, C, v, w, H_fc, Hff, vf, Efd)

  ### 4: apply retraction ###

  poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
  disps = disp_retr(disps, dz.view(B, -1, ht, wd), kx)

  intrinsics = focal_retr(intrinsics, df * (intrinsics[0, 0, 2] * 2))

  disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
  disps = disps.clamp(min=1e-3)

  return poses, disps, intrinsics


def BA_train(
    target,
    weight,
    eta,
    poses,
    disps,
    disps_sens,
    intrinsics,
    ii,
    jj,
    mot_prob=None,
    reg_type=None,
    fixedp=1,
    rig=1,
):
  """Full Bundle Adjustment"""
  B, P, ht, wd = disps.shape
  N = ii.shape[0]
  D = poses.manifold_dim

  # align estimated dispairty with mono-dispairty with precomputed scale
  disps = disps.clamp(min=1e-2, max=10.0)
  disps_sens = disps_sens.clamp(min=1e-2, max=10.0)

  if reg_type == 'log':
    log_diff = torch.log(disps_sens) - torch.log(disps)
    align_scale, _ = torch.median(log_diff.view(P, -1), dim=-1)  # .detach()
    align_scale = align_scale  # .detach()
  elif reg_type == 'linear':
    ratio = disps_sens / disps  # .clamp(min=1e-3)
    align_scale, _ = torch.median(ratio.view(P, -1), dim=-1)
    align_scale = align_scale  # .detach()

  # print("align_scale ", align_scale)

  ### 1: commpute jacobians and residuals ###
  coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
      poses, disps, intrinsics, ii, jj, jacobian=True
  )

  r = (target - coords).view(B, N, -1, 1)  # .double()
  w = 0.001 * (valid * weight).view(B, N, -1, 1)  # .double()

  # cost = torch.sum((valid * weight) * (target - coords)**2)/torch.sum(valid * weight)
  # print("wo focal residual ", cost)

  ### 2: construct linear system ###
  Ji = Ji.reshape(B, N, -1, D)  # .double()
  Jj = Jj.reshape(B, N, -1, D)  # .double()
  wJiT = (w * Ji).transpose(2, 3)
  wJjT = (w * Jj).transpose(2, 3)

  Jz = Jz.reshape(B, N, ht * wd, -1)

  Hii = torch.matmul(wJiT, Ji)
  Hij = torch.matmul(wJiT, Jj)
  # Hji = torch.matmul(wJjT, Ji)
  Hji = Hij.transpose(-2, -1)  # torch.matmul(wJjT, Ji)
  Hjj = torch.matmul(wJjT, Jj)

  vi = torch.matmul(wJiT, r).squeeze(-1)
  vj = torch.matmul(wJjT, r).squeeze(-1)

  Ei = (wJiT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
  Ej = (wJjT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)

  w = w.view(B, N, ht * wd, -1)
  r = r.view(B, N, ht * wd, -1)
  wk = torch.sum(w * r * Jz, dim=-1)
  Ck = torch.sum(w * Jz * Jz, dim=-1)

  kx, kk = torch.unique(ii, return_inverse=True)
  M = kx.shape[0]

  # only optimize keyframe poses
  P = P // rig - fixedp
  ii = ii // rig - fixedp
  jj = jj // rig - fixedp

  H = (
      safe_scatter_add_mat(Hii, ii, ii, P, P)
      + safe_scatter_add_mat(Hij, ii, jj, P, P)
      + safe_scatter_add_mat(Hji, jj, ii, P, P)
      + safe_scatter_add_mat(Hjj, jj, jj, P, P)
  )

  E = safe_scatter_add_mat(Ei, ii, kk, P, M) + safe_scatter_add_mat(
      Ej, jj, kk, P, M
  )

  v = safe_scatter_add_vec(vi, ii, P) + safe_scatter_add_vec(vj, jj, P)

  C = safe_scatter_add_vec(Ck, kk, M)
  w = safe_scatter_add_vec(wk, kk, M)

  # breakpoint()

  # why we can do direct addition? Because we only save diagonal non-zero entry!
  # This is for disparity L2 loss
  if reg_type == 'log':
    # This is for log disaprity L2 loss, which is perfectly scale invariance!!!
    m = mot_prob.float().view(
        *C.shape
    )  # (disps_sens[:, kx, ...] > 1e-8).float().view(*C.shape)

    alpha = 0.01
    J_reg = (1.0 / disps[:, kx]).view(*C.shape)
    C = C + m * alpha * J_reg + (1 - m) * (eta.view(*C.shape)) + 1e-7
    w = w - m * alpha * J_reg * (
        torch.log(disps[:, kx])
        + align_scale[None, :, None, None]
        - torch.log(disps_sens[:, kx])
    ).view(*w.shape)
    # print(torch.mean(torch.abs((torch.log(disps[:, kx]) + align_scale[None, :, None, None] - torch.log(disps_sens[:, kx])).view(*w.shape))))
  elif reg_type == 'linear':
    # This is for disparity L2 loss, which is scale aligned
    m = mot_prob.float().view(
        *C.shape
    )  # (disps_sens[:, kx, ...] > 1e-8).float().view(*C.shape)

    alpha = 0.05
    J_reg = 1.0
    C = C + m * alpha * J_reg + (1 - m) * (eta.view(*C.shape)) + 1e-7
    w = w - m * alpha * J_reg * (
        disps[:, kx] * align_scale[None, :, None, None] - disps_sens[:, kx]
    ).view(*w.shape)
    # print(torch.mean(torch.abs((disps[:, kx] * align_scale[None, :, None, None] - disps_sens[:, kx]).view(*w.shape))))
  else:
    # This is Vanila version!!
    C = C + eta.view(*C.shape) + 1e-7

  H = H.view(B, P, P, D, D)
  E = E.view(B, P, M, D, ht * wd)

  ### 3: solve the system ###
  # breakpoint()
  dx, dz = schur_solve(H, E, C, v, w)

  ### 4: apply retraction ###
  poses = pose_retr(poses, dx.float(), torch.arange(P) + fixedp)
  disps = disp_retr(disps, dz.float().view(B, -1, ht, wd), kx)

  disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
  disps = disps.clamp(min=1e-3)

  return poses, disps


def MoBA(
    target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1
):
  """Motion only bundle adjustment"""

  B, P, ht, wd = disps.shape
  N = ii.shape[0]
  D = poses.manifold_dim

  ### 1: commpute jacobians and residuals ###
  coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
      poses, disps, intrinsics, ii, jj, jacobian=True
  )

  r = (target - coords).view(B, N, -1, 1)
  w = 0.001 * (valid * weight).view(B, N, -1, 1)

  ### 2: construct linear system ###
  Ji = Ji.reshape(B, N, -1, D)
  Jj = Jj.reshape(B, N, -1, D)
  wJiT = (w * Ji).transpose(2, 3)
  wJjT = (w * Jj).transpose(2, 3)

  Hii = torch.matmul(wJiT, Ji)
  Hij = torch.matmul(wJiT, Jj)
  Hji = torch.matmul(wJjT, Ji)
  Hjj = torch.matmul(wJjT, Jj)

  vi = torch.matmul(wJiT, r).squeeze(-1)
  vj = torch.matmul(wJjT, r).squeeze(-1)

  # only optimize keyframe poses
  P = P // rig - fixedp
  ii = ii // rig - fixedp
  jj = jj // rig - fixedp

  H = (
      safe_scatter_add_mat(Hii, ii, ii, P, P)
      + safe_scatter_add_mat(Hij, ii, jj, P, P)
      + safe_scatter_add_mat(Hji, jj, ii, P, P)
      + safe_scatter_add_mat(Hjj, jj, jj, P, P)
  )

  v = safe_scatter_add_vec(vi, ii, P) + safe_scatter_add_vec(vj, jj, P)

  H = H.view(B, P, P, D, D)

  ### 3: solve the system ###
  dx = block_solve(H, v)

  ### 4: apply retraction ###
  poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
  return poses
