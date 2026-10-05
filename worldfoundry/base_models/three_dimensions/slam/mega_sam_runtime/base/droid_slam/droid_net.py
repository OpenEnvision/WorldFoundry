from collections import OrderedDict

from geom.ba import BA_f_train, BA_train
from geom.graph_utils import graph_to_edge_list, keyframe_indicies
import geom.projective_ops as pops
from lietorch import SE3
from modules.clipping import GradientClip
from modules.corr import CorrBlock
from modules.extractor import BasicEncoder
from modules.gru import ConvGRU
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.scatter import scatter_mean


def cvx_upsample(data, mask):
  """upsample pixel-wise transformation field"""
  batch, ht, wd, dim = data.shape
  data = data.permute(0, 3, 1, 2)
  mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
  mask = torch.softmax(mask, dim=2)

  up_data = F.unfold(data, [3, 3], padding=1)
  up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

  up_data = torch.sum(mask * up_data, dim=2)
  up_data = up_data.permute(0, 4, 2, 5, 3, 1)
  up_data = up_data.reshape(batch, 8 * ht, 8 * wd, dim)

  return up_data


def upsample_disp(disp, mask):
  batch, num, ht, wd = disp.shape
  disp = disp.view(batch * num, ht, wd, 1)
  mask = mask.view(batch * num, -1, ht, wd)
  return cvx_upsample(disp, mask).view(batch, num, 8 * ht, 8 * wd)


class GraphAgg(nn.Module):

  def __init__(self):
    super(GraphAgg, self).__init__()
    self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
    self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
    self.relu = nn.ReLU(inplace=True)

    self.eta = nn.Sequential(
        nn.Conv2d(128, 1, 3, padding=1), GradientClip(), nn.Softplus()
    )

    self.upmask = nn.Sequential(nn.Conv2d(128, 8 * 8 * 9, 1, padding=0))

  def forward(self, net, ii):
    batch, num, ch, ht, wd = net.shape
    net = net.view(batch * num, ch, ht, wd)

    _, ix = torch.unique(ii, return_inverse=True)
    net = self.relu(self.conv1(net))

    net = net.view(batch, num, 128, ht, wd)
    net = scatter_mean(net, ix, dim=1)
    net = net.view(-1, 128, ht, wd)

    net = self.relu(self.conv2(net))

    eta = self.eta(net).view(batch, -1, ht, wd)
    upmask = self.upmask(net).view(batch, -1, 8 * 8 * 9, ht, wd)

    return 0.01 * eta, upmask


class LayerNorm(nn.Module):
  r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.

  The ordering of the dimensions in the inputs. channels_last corresponds to
  inputs with shape (batch_size, height, width, channels) while channels_first
  corresponds to inputs with shape (batch_size, channels, height, width).
  """

  def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(normalized_shape))
    self.bias = nn.Parameter(torch.zeros(normalized_shape))
    self.eps = eps
    self.data_format = data_format
    if self.data_format not in ["channels_last", "channels_first"]:
      raise NotImplementedError
    self.normalized_shape = (normalized_shape,)

  def forward(self, x):
    if self.data_format == "channels_last":
      return F.layer_norm(
          x.permute(0, 2, 3, 1),
          self.normalized_shape,
          self.weight,
          self.bias,
          self.eps,
      ).permute(0, 3, 1, 2)
    elif self.data_format == "channels_first":
      u = x.mean(1, keepdim=True)
      s = (x - u).pow(2).mean(1, keepdim=True)
      x = (x - u) / torch.sqrt(s + self.eps)
      x = self.weight[:, None, None] * x + self.bias[:, None, None]
      return x


class GraphAggMotion(nn.Module):  # OLD VERSION

  def __init__(self):
    super(GraphAggMotion, self).__init__()
    self.dim = 128
    self.conv0 = nn.Conv2d(256 + 2, self.dim, 3, padding=1)
    self.conv1 = nn.Conv2d(self.dim, self.dim, 3, padding=1)

    self.conv3 = nn.Conv2d(self.dim, self.dim, 3, padding=1)
    self.conv4 = nn.Conv2d(self.dim + 1, self.dim + 1, 3, padding=1)
    self.conv5 = nn.Conv2d(self.dim, self.dim, 3, padding=1)

    self.norm_0 = LayerNorm(self.dim)
    self.norm_1 = LayerNorm(self.dim)
    self.norm_3 = LayerNorm(self.dim)
    self.norm_4 = LayerNorm(self.dim + 1)
    self.norm_5 = LayerNorm(self.dim)

    self.relu0 = nn.GELU()
    self.relu1 = nn.GELU()
    self.relu3 = nn.GELU()
    self.relu4 = nn.GELU()
    self.relu5 = nn.GELU()

    self.eta = nn.Sequential(
        nn.Conv2d(self.dim + 1, self.dim, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(self.dim, 1, 3, padding=1),
    )
    # GradientClip(),
    # nn.Softplus())

    self.eta[-1].bias.data.fill_(0.0)
    self.eta[-1].weight.data.fill_(0.0)

    self.logits_2 = nn.Sequential(
        # nn.Conv2d(256, 128, 3, padding=1),
        nn.Conv2d(self.dim, self.dim, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(self.dim, 1, 3, padding=1),
        nn.Sigmoid(),
    )
    # nn.Tanh())

    self.logits_2[-2].bias.data.fill_(0.0)
    self.logits_2[-2].weight.data.fill_(0.0)

    # self.upmask_2 = nn.Sequential(
    # nn.Conv2d(256, 128, 3, padding=1),
    # nn.ReLU(inplace=True),
    # nn.Conv2d(256, 8*8*9, 3, padding=1))

    self.add_residual = False

  def forward(self, net_, ii, init_w, init_eta, disps_sensor):
    batch, num, ch, ht, wd = net_.shape
    net_0 = net_.view(batch * num, ch, ht, wd)
    _, ix = torch.unique(ii, return_inverse=True)

    # disps_sensor_ = torch.index_select(disps_sensor, dim=1, index=ii.long())[..., None].permute(0, 1, 4, 2, 3).squeeze(0)
    avg_motion = torch.mean(net_0.view(batch * num, ch, ht * wd), axis=-1)[
        ..., None, None
    ].expand(-1, -1, ht, wd)

    init_w_ = init_w.squeeze(0).permute(0, 3, 1, 2)
    # net_0_ = torch.cat([net_0, disps_sensor_, avg_motion, init_w_], dim=1)
    net_0_ = torch.cat([net_0, avg_motion, init_w_], dim=1)

    net_1 = self.relu0(self.norm_0(self.conv0(net_0_)))
    net_2 = self.relu1(self.norm_1(self.conv1(net_1))) + net_1
    net_2 = self.relu3(self.norm_3(self.conv3(net_2))) + net_2

    net_2 = net_2.view(batch, num, self.dim, ht, wd)
    net = scatter_mean(net_2, ix, dim=1)

    if self.add_residual:
      raise NotImplementedError
      mot_prob = torch.nn.functional.relu(self.logits_2(net_5) + init_w_)
    else:
      net_5_avg = self.relu5(self.norm_5(self.conv5(net.squeeze(0))))[None, ...]
      net_5 = (
          net_2 + torch.index_select(net_5_avg, dim=1, index=ix.long())
      ).squeeze(0)
      refined_w = self.logits_2(net_5)
      mot_prob = refined_w * init_w_
      # mot_prob = (self.logits_2(net_5) + 1.) * init_w_

    mot_prob = (
        mot_prob.view(batch, num, 2, ht, wd).permute(0, 1, 3, 4, 2).contiguous()
    )
    # upmask_m = self.upmask_2(net_5).view(batch, -1, 8*8*9, ht, wd)
    # upmask_m = None

    # net_2 = net_2.view(batch, num, self.dim, ht, wd)
    # net = scatter_mean(net_2, ix, dim=1)
    net_3 = net.view(-1, self.dim, ht, wd)
    net_3_ = torch.cat([net_3, init_eta.permute(1, 0, 2, 3)], dim=1)
    net_4 = self.relu4(self.norm_4(self.conv4(net_3_))) + net_3_

    # new_eta = init_eta + torch.nn.functional.softplus(self.eta(net_4).view(batch, -1, ht, wd) - 6.)
    new_eta = init_eta + torch.exp(
        self.eta(net_4).view(batch, -1, ht, wd) - 10.0
    )

    return new_eta, mot_prob, refined_w


class UpdateModule(nn.Module):

  def __init__(self):
    super(UpdateModule, self).__init__()
    cor_planes = 4 * (2 * 3 + 1) ** 2

    self.corr_encoder = nn.Sequential(
        nn.Conv2d(cor_planes, 128, 1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.ReLU(inplace=True),
    )

    self.flow_encoder = nn.Sequential(
        nn.Conv2d(4, 128, 7, padding=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, 3, padding=1),
        nn.ReLU(inplace=True),
    )

    self.weight = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 2, 3, padding=1),
        GradientClip(),
        nn.Sigmoid(),
    )

    self.delta = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 2, 3, padding=1),
        GradientClip(),
    )

    self.gru = ConvGRU(128, 128 + 128 + 64)

    self.agg = GraphAgg()
    self.agg_motion = GraphAggMotion()

  def forward(
      self,
      net,
      inp,
      corr,
      flow=None,
      ii=None,
      jj=None,
      disps=None,
      disps_sensor=None,
  ):
    """RaftSLAM update operator"""

    batch, num, ch, ht, wd = net.shape

    if flow is None:
      flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

    output_dim = (batch, num, -1, ht, wd)
    net = net.view(batch * num, -1, ht, wd)
    inp = inp.view(batch * num, -1, ht, wd)
    corr = corr.view(batch * num, -1, ht, wd)
    flow = flow.view(batch * num, -1, ht, wd)

    corr = self.corr_encoder(corr)
    flow = self.flow_encoder(flow)
    net = self.gru(net, inp, corr, flow)

    ### update variables ###
    delta = self.delta(net).view(*output_dim)
    weight = self.weight(net).view(*output_dim)

    delta = delta.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
    weight = weight.permute(0, 1, 3, 4, 2)[..., :2].contiguous()

    net = net.view(*output_dim)

    if ii is not None:
      eta, upmask_d = self.agg(net, ii.to(net.device))

      mot_prob = None
      upmask_m = None
      new_eta, mot_prob, refined_w = self.agg_motion(
          net, ii.to(net.device), weight, eta, disps_sensor
      )

      return net, delta, weight, eta, upmask_d, mot_prob, refined_w
      # return net, delta, weight, new_eta, upmask_d, reg_prob, upmask_r, mot_prob, refined_w

    else:
      return net, delta, weight


class DroidNet(nn.Module):

  def __init__(self):
    super(DroidNet, self).__init__()
    self.fnet = BasicEncoder(output_dim=128, norm_fn="instance")
    self.cnet = BasicEncoder(output_dim=256, norm_fn="none")
    self.update = UpdateModule()
    # self.multi_view_u = multi_view_u
    # assert multi_view_u == False

    # self.reg_type = reg_type
    # print("=========== multi_view_u ", multi_view_u)
    # print("=========== reg_type ", reg_type)

  def extract_features(self, images):
    """run feeature extraction networks"""
    # normalize images, to rgb from bgr!
    images = images[:, :, [2, 1, 0]] / 255.0
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
    images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

    fmaps = self.fnet(images)
    net = self.cnet(images)

    net, inp = net.split([128, 128], dim=2)
    net = torch.tanh(net)
    inp = torch.relu(inp)
    return fmaps, net, inp

  def forward(
      self,
      Gs,
      images,
      disps,
      disps_sensor,
      intrinsics,
      graph=None,
      num_steps=12,
      fixedp=2,
      motion_only=False,
  ):
    """Estimates SE3 or Sim3 between pair of frames"""
    u = keyframe_indicies(graph)
    ii, jj, kk = graph_to_edge_list(graph)

    ii = ii.to(device=images.device, dtype=torch.long)
    jj = jj.to(device=images.device, dtype=torch.long)

    fmaps, net, inp = self.extract_features(images)

    net, inp = net[:, ii], inp[:, ii]

    corr_fn = CorrBlock(fmaps[:, ii], fmaps[:, jj], num_levels=4, radius=3)

    ht, wd = images.shape[-2:]
    coords0 = pops.coords_grid(ht // 8, wd // 8, device=images.device)

    coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
    target = coords1.clone()

    Gs_list, disp_list, residual_list = [], [], []
    intrinsic_list = []
    mot_mask_list = []
    for step in range(num_steps):
      Gs = Gs.detach()
      disps = disps.detach()
      coords1 = coords1.detach()
      target = target.detach()

      # extract motion features
      corr = corr_fn(coords1)
      resd = target - coords1
      flow = coords1 - coords0

      motion = torch.cat([flow, resd], dim=-1)
      motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

      net, delta, weight, eta, upmask_d, mot_prob, upmask_m = self.update(
          net, inp, corr, motion, ii, jj, disps, disps_sensor
      )

      target = coords1 + delta

      weight_ = weight

      # Gs_init = Gs.detach()
      # disps_init = disps.clone()
      intrinsics_init = intrinsics.clone()

      for i in range(2):
        Gs, disps, intrinsics = BA_f_train(
            target,
            weight_,
            eta,
            Gs,
            disps,
            disps_sensor,
            intrinsics,
            ii,
            jj,
            fixedp=fixedp,
        )

      # Gs = Gs_init.detach()
      # disps = disps_init.clone()
      # intrinsics = intrinsics_init.clone()

      # for i in range(2):
      #     Gs, disps = BA_train(target, weight_, eta, Gs, disps,
      #             disps_sensor, intrinsics, ii, jj,
      #             reg_type=self.reg_type, fixedp=fixedp)

      # breakpoint()
      print("intrinsics ", intrinsics_init, intrinsics)

      coords1, valid_mask = pops.projective_transform(
          Gs, disps, intrinsics, ii, jj
      )
      residual = target - coords1

      Gs_list.append(Gs)
      disp_list.append(upsample_disp(disps, upmask_d))
      residual_list.append(valid_mask * residual)
      intrinsic_list.append(intrinsics + 0.0)
      mot_mask_list.append(torch.clamp(weight_, 0.0, 1.0))

    return Gs_list, disp_list, residual_list, mot_mask_list, intrinsic_list
