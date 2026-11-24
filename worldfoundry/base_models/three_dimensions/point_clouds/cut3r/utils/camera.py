import torch
import torch.nn as nn

from worldfoundry.core.geometry.transforms import quaternion_wxyz_to_rotation_matrix as quaternion_to_matrix
from worldfoundry.core.geometry.transforms import rotation_matrix_to_unit_quaternion_wxyz as matrix_to_quaternion

from ..blocks import Mlp
from ..heads.postprocess import postprocess_pose

inf = float("inf")


class PoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        mlp_ratio=4,
        pose_encoding_type="absT_quaR",
    ):
        super().__init__()

        self.pose_encoding_type = pose_encoding_type
        if self.pose_encoding_type == "absT_quaR":
            self.target_dim = 7

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=self.target_dim,
            drop=0,
        )

    def forward(
        self,
        pose_feat,
    ):
        """
        pose_feat: BxC
        preliminary_cameras: cameras in opencv coordinate.
        """

        pred_cameras = self.mlp(pose_feat)  # Bx7, 3 for absT, 4 for quaR
        return pred_cameras


class PoseEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        mlp_ratio=4,
        pose_mode=("exp", -inf, inf),
        pose_encoding_type="absT_quaR",
    ):
        super().__init__()
        self.pose_encoding_type = pose_encoding_type
        self.pose_mode = pose_mode

        if self.pose_encoding_type == "absT_quaR":
            self.target_dim = 7

        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim,
            out_dim=hidden_size,
            n_harmonic_functions=10,
            append_input=True,
        )
        self.pose_encoder = Mlp(
            in_features=self.embed_pose.out_dim,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            drop=0,
        )

    def forward(self, camera):
        pose_enc = camera_to_pose_encoding(
            camera,
            pose_encoding_type=self.pose_encoding_type,
        ).to(camera.dtype)
        pose_enc = postprocess_pose(pose_enc, self.pose_mode, inverse=True)
        pose_feat = self.embed_pose(pose_enc)
        pose_feat = self.pose_encoder(pose_feat)
        return pose_feat


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        The harmonic embedding layer supports the classical
        Nerf positional encoding described in
        `NeRF <https://arxiv.org/abs/2003.08934>`_
        and the integrated position encoding in
        `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.

        During the inference you can provide the extra argument `diag_cov`.

        If `diag_cov is None`, it converts
        rays parametrized with a `ray_bundle` to 3D points by
        extending each ray according to the corresponding length.
        Then it converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]

        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.


        If `diag_cov is not None`, it approximates
        conical frustums following a ray bundle as gaussians,
        defined by x, the means of the gaussians and diag_cov,
        the diagonal covariances.
        Then it converts each gaussian
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                sin(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),
                ...
                sin(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                cos(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                cos(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),,
                ...
                cos(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                x[..., i],              # only present if append_input is True.
            ]

        where N equals `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.

        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (embed.sin(), embed.cos(), x)
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(n_harmonic_functions, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
        self.register_buffer(
            "_zero_half_pi",
            torch.tensor([0.0, 0.5 * torch.pi]),
            persistent=False,
        )
        self.append_input = append_input

    def forward(self, x: torch.Tensor, diag_cov: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
            diag_cov: An optional tensor of shape `(..., dim)`
                representing the diagonal covariance matrices of our Gaussians, joined with x
                as means of the Gaussians.

        Returns:
            embedding: a harmonic embedding of `x` of shape
            [..., (n_harmonic_functions * 2 + int(append_input)) * num_points_per_ray]
        """

        embed = x[..., None] * self._frequencies

        embed = embed[..., None, :, :] + self._zero_half_pi[..., None, None]

        embed = embed.sin()
        if diag_cov is not None:
            x_var = diag_cov[..., None] * torch.pow(self._frequencies, 2)
            exp_var = torch.exp(-0.5 * x_var)

            embed = embed * exp_var[..., None, :, :]

        embed = embed.reshape(*x.shape[:-1], -1)

        if self.append_input:
            return torch.cat([embed, x], dim=-1)
        return embed

    @staticmethod
    def get_output_dim_static(input_dims: int, n_harmonic_functions: int, append_input: bool) -> int:
        """
        Utility to help predict the shape of the output of `forward`.

        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(input_dims, len(self._frequencies), self.append_input)


class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, out_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(n_harmonic_functions=n_harmonic_functions, append_input=append_input)

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding


def camera_to_pose_encoding(
    camera,
    pose_encoding_type="absT_quaR",
):
    """
    Inverse to pose_encoding_to_camera
    camera: opencv, cam2world
    """
    if pose_encoding_type == "absT_quaR":
        quaternion_R = matrix_to_quaternion(camera[:, :3, :3])

        pose_encoding = torch.cat([camera[:, :3, 3], quaternion_R], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding


def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR",
):
    """
    Args:
        pose_encoding: A tensor of shape `BxC`, containing a batch of
                        `B` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
    """

    if pose_encoding_type == "absT_quaR":
        abs_T = pose_encoding[:, :3]
        quaternion_R = pose_encoding[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    c2w_mats = torch.eye(4, 4).to(R.dtype).to(R.device)
    c2w_mats = c2w_mats[None].repeat(len(R), 1, 1)
    c2w_mats[:, :3, :3] = R
    c2w_mats[:, :3, 3] = abs_T

    return c2w_mats


def quaternion_conjugate(q):
    """Compute the conjugate of quaternion q (w, x, y, z)."""

    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return q_conj


def quaternion_multiply(q1, q2):
    """Multiply two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    q_vec = q[..., 1:]
    q_w = q[..., :1]

    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    v_rot = v + q_w * t + torch.cross(q_vec, t, dim=-1)
    return v_rot


def relative_pose_absT_quatR(t1, q1, t2, q2):
    """Compute the relative translation and quaternion between two poses."""

    q1_inv = quaternion_conjugate(q1)

    q_rel = quaternion_multiply(q1_inv, q2)

    delta_t = t2 - t1
    t_rel = rotate_vector(q1_inv, delta_t)
    return t_rel, q_rel
