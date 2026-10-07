"""Grid-output DiT wrapper used by distilled π-Flow checkpoints."""

from __future__ import annotations

from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.dit import DiffusionTransformer3D

_MIN_GRID_POINTS = 2


class PiFlowDiffusionTransformer3D(DiffusionTransformer3D):
    """K6 DiT whose output head predicts a DX grid per modality."""

    def __init__(
        self,
        *,
        n_grid: int,
        out_visual_dim: int,
        out_audio_dim: int | None = None,
        **kwargs,
    ) -> None:
        if n_grid < _MIN_GRID_POINTS:
            raise ValueError(f"PiFlow n_grid must be >= 2, got {n_grid}")
        base_audio_dim = out_audio_dim or int(kwargs.get("in_audio_dim", 20))
        super().__init__(
            out_visual_dim=out_visual_dim * n_grid,
            out_audio_dim=base_audio_dim * n_grid,
            **kwargs,
        )
        self.n_grid = int(n_grid)
        self.dx_out_visual_dim = int(out_visual_dim)
        self.dx_out_audio_dim = int(base_audio_dim)
        self.piflow_params = {"n_grid": self.n_grid}

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        if not isinstance(result, tuple):
            raise RuntimeError("PiFlow DiT requires the fused video/audio forward path")
        video, audio = result
        video_shape = video.shape
        video = video.view(*video_shape[:-1], self.n_grid, self.dx_out_visual_dim)
        video = video.movedim(-2, 1)
        audio_shape = audio.shape
        audio = audio.view(*audio_shape[:-1], self.n_grid, self.dx_out_audio_dim)
        audio = audio.movedim(-2, 1)
        return video, audio
