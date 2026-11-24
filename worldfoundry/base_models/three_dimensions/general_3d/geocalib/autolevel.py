"""Auto-level camera estimation built on the in-tree GeoCalib backbone."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MSCAN, ConvModule, LightHamHead


def _build_rotation(pitch_deg: torch.Tensor, roll_deg: torch.Tensor) -> torch.Tensor:
    pitch = torch.deg2rad(pitch_deg)
    roll = torch.deg2rad(roll_deg)
    zero = torch.zeros_like(pitch)
    one = torch.ones_like(pitch)
    rotation_x = torch.stack(
        [
            torch.stack([one, zero, zero], dim=-1),
            torch.stack([zero, torch.cos(pitch), -torch.sin(pitch)], dim=-1),
            torch.stack([zero, torch.sin(pitch), torch.cos(pitch)], dim=-1),
        ],
        dim=-2,
    )
    rotation_z = torch.stack(
        [
            torch.stack([torch.cos(roll), -torch.sin(roll), zero], dim=-1),
            torch.stack([torch.sin(roll), torch.cos(roll), zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )
    return torch.bmm(rotation_z, rotation_x)


def perspective_to_erp_grid(
    fov_deg: torch.Tensor,
    pitch_deg: torch.Tensor,
    roll_deg: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Map perspective pixels to normalized equirectangular coordinates."""
    device = fov_deg.device
    batch_size = fov_deg.shape[0]
    aspect_ratio = width / float(height)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-aspect_ratio, aspect_ratio, width, device=device),
        indexing="ij",
    )
    grid_x = grid_x[None, :, :, None].expand(batch_size, -1, -1, -1)
    grid_y = grid_y[None, :, :, None].expand(batch_size, -1, -1, -1)
    focal = (1.0 / torch.tan(torch.deg2rad(fov_deg / 2.0)))[:, None, None, None]
    focal = focal.expand(batch_size, height, width, 1)
    rays = F.normalize(torch.cat([grid_x, grid_y, focal], dim=-1), dim=-1)
    rotation = _build_rotation(pitch_deg, roll_deg)
    rays = torch.bmm(rays.reshape(batch_size, -1, 3), rotation.transpose(1, 2))
    rays = rays.reshape(batch_size, height, width, 3)
    longitude = torch.atan2(rays[..., 0], rays[..., 2]) / math.pi
    latitude = torch.asin(rays[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / (math.pi / 2.0)
    return torch.stack([longitude, latitude], dim=1)


def preprocess_image(
    image: torch.Tensor,
    short_side: int = 320,
    divisor: int = 32,
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Resize the short side and center-crop both dimensions to a divisor."""
    batched = image.ndim == 4
    if not batched:
        image = image.unsqueeze(0)
    _, _, height, width = image.shape
    if height <= width:
        new_height = short_side
        new_width = round(short_side * width / height)
    else:
        new_width = short_side
        new_height = round(short_side * height / width)
    resized = F.interpolate(
        image,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    crop_height = (new_height // divisor) * divisor
    crop_width = (new_width // divisor) * divisor
    top = (new_height - crop_height) // 2
    left = (new_width - crop_width) // 2
    cropped = resized[:, :, top : top + crop_height, left : left + crop_width]
    if not batched:
        cropped = cropped.squeeze(0)
    return cropped, (new_width / width, new_height / height)


class LowLevelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvModule(3, 64, kernel_size=3, padding=1)
        self.conv2 = ConvModule(64, 64, kernel_size=3, padding=1)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image = data["image"]
        if image.shape[-1] % 32 or image.shape[-2] % 32:
            raise ValueError("AutoLevel input dimensions must be divisible by 32")
        return {"features": self.conv2(self.conv1(image))}


def _ray_position_encoding(height: int, width: int, device: torch.device) -> torch.Tensor:
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x * width / float(height), grid_y], dim=0).unsqueeze(0)


class FlowDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ham_head = LightHamHead()
        del self.ham_head.linear_pred_uncertainty
        self.ray_pe_proj = nn.Conv2d(
            self.ham_head.ham_channels + 2,
            self.ham_head.ham_channels,
            kernel_size=1,
        )
        self.linear_pred_flow = nn.Conv2d(
            self.ham_head.out_channels + 2,
            2,
            kernel_size=1,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        ray_pe: torch.Tensor,
    ) -> torch.Tensor:
        head = self.ham_head
        levels = [features["hl"][index] for index in head.in_index]
        levels = [
            F.interpolate(
                level,
                size=levels[0].shape[2:],
                mode="bilinear",
                align_corners=head.align_corners,
            )
            for level in levels
        ]
        hidden = head.squeeze(torch.cat(levels, dim=1))
        ray_pe_small = F.interpolate(
            ray_pe,
            size=hidden.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.ray_pe_proj(torch.cat([hidden, ray_pe_small], dim=1))
        features_out = head.align(head.hamburger(hidden))
        features_out = F.interpolate(features_out, scale_factor=2, mode="bilinear", align_corners=False)
        features_out = head.out_conv(features_out)
        features_out = F.interpolate(features_out, scale_factor=2, mode="bilinear", align_corners=False)
        features_out = head.ll_fusion(features_out, features["ll"].clone())
        return self.linear_pred_flow(torch.cat([features_out, ray_pe], dim=1))


def _camera_pixel_uv(
    parameters: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    log_fov, pitch, roll = parameters
    focal = 1.0 / torch.tan(torch.exp(log_fov) * (math.pi / 180.0) / 2.0)
    ray = F.normalize(
        torch.stack([grid_x.to(focal.dtype), grid_y.to(focal.dtype), focal]),
        dim=0,
    )
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
    zero, one = cos_pitch * 0.0, cos_pitch * 0.0 + 1.0
    rotation_x = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cos_pitch, -sin_pitch]),
            torch.stack([zero, sin_pitch, cos_pitch]),
        ]
    )
    rotation_z = torch.stack(
        [
            torch.stack([cos_roll, -sin_roll, zero]),
            torch.stack([sin_roll, cos_roll, zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    direction = rotation_z @ rotation_x @ ray
    return torch.stack(
        [
            torch.atan2(direction[0], direction[2]) / math.pi,
            torch.asin(direction[1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / (math.pi / 2.0),
        ]
    )


class RigidFilter(nn.Module):
    def __init__(
        self,
        lm_steps: int = 10,
        lm_lambda: float = 0.1,
        fit_res: int = 128,
        **_,
    ):
        super().__init__()
        self.lm_steps = lm_steps
        self.lm_lambda = lm_lambda
        self.fit_res = fit_res

    @staticmethod
    @torch.no_grad()
    def _initial_parameters(
        flow: torch.Tensor,
        aspect: float,
        device: torch.device,
    ) -> torch.Tensor:
        height, width = flow.shape[1:]
        longitude = flow[0].flatten() * math.pi
        latitude = flow[1].flatten() * (math.pi / 2.0)
        cos_latitude = torch.cos(latitude)
        world_rays = torch.stack(
            [
                torch.sin(longitude) * cos_latitude,
                torch.sin(latitude),
                torch.cos(longitude) * cos_latitude,
            ],
            dim=-1,
        )
        vertical_span = float((flow[1].max() - flow[1].min()).clamp(0.05, 1.98))
        fov_estimate = max(10.0, min(vertical_span * 90.0, 150.0))
        focal_estimate = 1.0 / math.tan(math.radians(fov_estimate / 2.0))
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-aspect, aspect, width, device=device),
            indexing="ij",
        )
        camera_rays = F.normalize(
            torch.stack([grid_x, grid_y, torch.full_like(grid_x, focal_estimate)], dim=-1).reshape(-1, 3),
            dim=-1,
        )
        moment = camera_rays.T @ world_rays
        try:
            left, _, right_transpose = torch.linalg.svd(moment)
            orientation = right_transpose.T @ left.T
            determinant = torch.linalg.det(orientation).item()
            correction = torch.diag(torch.tensor([1.0, 1.0, determinant], device=device))
            rotation = right_transpose.T @ correction @ left.T
        except RuntimeError:
            rotation = torch.eye(3, device=device)
        pitch = torch.asin(rotation[2, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        roll = torch.atan2(rotation[1, 0], rotation[0, 0])

        fov_candidates = torch.linspace(10.0, 150.0, 1401, device=device)
        focal_candidates = 1.0 / torch.tan(torch.deg2rad(fov_candidates / 2.0))
        grid_x_flat, grid_y_flat = grid_x.flatten(), grid_y.flatten()
        candidate_rays = F.normalize(
            torch.stack(
                [
                    grid_x_flat[None].expand(1401, -1),
                    grid_y_flat[None].expand(1401, -1),
                    focal_candidates[:, None].expand(-1, grid_x_flat.numel()),
                ],
                dim=-1,
            ),
            dim=-1,
        )
        errors = ((candidate_rays @ rotation.T - world_rays[None]) ** 2).sum(dim=(-1, -2))
        return torch.stack([torch.log(fov_candidates[errors.argmin()]), pitch, roll])

    @staticmethod
    def _lm_step(
        parameters: torch.Tensor,
        target: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        damping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch.func import jacrev, vmap

        grid_x_flat, grid_y_flat = grid_x.flatten(), grid_y.flatten()
        jacobian = vmap(lambda x, y: jacrev(lambda p: _camera_pixel_uv(p, x, y))(parameters))(grid_x_flat, grid_y_flat)
        residual = vmap(lambda x, y: _camera_pixel_uv(parameters, x, y))(grid_x_flat, grid_y_flat)
        residual = residual - target.permute(1, 2, 0).reshape(-1, 2)
        hessian = torch.einsum("nji,njk->ik", jacobian, jacobian)
        gradient = torch.einsum("nji,nj->i", jacobian, residual)
        damped = hessian + (hessian.diagonal() * damping).clamp(min=1e-6).diag()
        try:
            cholesky = torch.linalg.cholesky(damped.cpu())
            delta = torch.cholesky_solve(gradient.cpu()[:, None], cholesky)
            delta = delta.squeeze(-1).to(parameters.device)
        except RuntimeError:
            delta = torch.zeros_like(gradient)
        candidate = parameters - delta
        candidate = torch.stack(
            [
                candidate[0].clamp(math.log(10.0), math.log(150.0)),
                candidate[1],
                candidate[2],
            ]
        )
        return candidate, (residual**2).sum()

    def _fit_one(
        self,
        flow: torch.Tensor,
        aspect: float,
        device: torch.device,
    ) -> torch.Tensor:
        height, width = flow.shape[1:]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-aspect, aspect, width, device=device),
            indexing="ij",
        )
        parameters = self._initial_parameters(flow, aspect, device)
        damping = flow.new_tensor(self.lm_lambda)
        previous_cost = None
        for _ in range(self.lm_steps):
            parameters, cost = self._lm_step(parameters, flow, grid_x, grid_y, damping)
            if previous_cost is not None:
                damping = torch.where(cost > previous_cost, damping * 10.0, damping * 0.1)
                damping = damping.clamp(1e-6, 1e2)
            previous_cost = cost
        return parameters

    def forward(
        self,
        dense_grid: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device, dtype = dense_grid.device, dense_grid.dtype
        aspect = width / float(height)
        fit_width = self.fit_res if aspect >= 1.0 else max(int(self.fit_res * aspect), 1)
        fit_height = self.fit_res if aspect < 1.0 else max(int(self.fit_res / aspect), 1)
        flow = F.interpolate(
            dense_grid.detach().float(),
            size=(fit_height, fit_width),
            mode="bilinear",
            align_corners=False,
        )
        parameters = [self._fit_one(item, aspect, device) for item in flow]
        parameters = torch.stack(parameters)
        fov = torch.exp(parameters[:, 0])
        pitch = torch.rad2deg(parameters[:, 1])
        roll = torch.rad2deg(parameters[:, 2])
        rigid_grid = perspective_to_erp_grid(fov, pitch, roll, height, width)
        return rigid_grid.to(dtype), torch.stack([fov, pitch, roll], dim=1).to(dtype)


class FlowEstimator(nn.Module):
    """Inference-only single-image camera leveling model."""

    def __init__(self, rigid_filter_cfg: dict | None = None):
        super().__init__()
        self.backbone = MSCAN()
        self.ll_enc = LowLevelEncoder()
        self.decoder = FlowDecoder()
        self.rigid_filter = RigidFilter(**(rigid_filter_cfg or {}))
        self.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = image.shape
        high_level = self.backbone({"image": image})["features"]
        low_level = self.ll_enc({"image": image})["features"]
        ray_pe = _ray_position_encoding(height, width, image.device)
        ray_pe = ray_pe.expand(batch_size, -1, -1, -1).to(image.dtype)
        delta = self.decoder({"hl": high_level, "ll": low_level}, ray_pe)
        base_fov = torch.full((batch_size,), 60.0, device=image.device)
        zero = torch.zeros(batch_size, device=image.device)
        base_grid = perspective_to_erp_grid(base_fov, zero, zero, height, width)
        dense_grid = (base_grid + torch.tanh(delta) * 2.0).clamp(-1.0, 1.0)
        with torch.autocast(device_type=image.device.type, enabled=False):
            rigid_grid, parameters = self.rigid_filter(dense_grid.float(), height, width)
        return {
            "dense_grid": dense_grid,
            "rigid_grid": rigid_grid.to(dense_grid.dtype),
            "params": parameters.to(dense_grid.dtype),
        }
