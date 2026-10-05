import copy

import pytest
import torch

from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts


class _MixedDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(4, 8, 3, padding=1)
        self.conv2d = torch.nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, value):
        value = torch.nn.functional.silu(self.conv3d(value))
        b, c, t, h, w = value.shape
        frames = value.transpose(1, 2).reshape(b * t, c, h, w)
        return self.conv2d(frames).reshape(b, t, 3, h, w)


def test_mixed_layout_preserves_parameters_weights_and_pixels():
    torch.manual_seed(81)
    reference = _MixedDecoder().eval()
    candidate = copy.deepcopy(reference)
    parameters = dict(candidate.named_parameters())
    values = {key: value.clone() for key, value in candidate.state_dict().items()}
    report = convert_convolution_weight_layouts(candidate, conv2d=True, conv3d=True)
    assert report.conv2d_total == report.conv2d_converted == 1
    assert report.conv3d_total == report.conv3d_converted == 1
    for name, value in candidate.named_parameters():
        assert value is parameters[name]
        torch.testing.assert_close(value, values[name], rtol=0, atol=0)
    with torch.no_grad():
        value = torch.randn(2, 4, 3, 7, 9)
        torch.testing.assert_close(candidate(value), reference(value), rtol=1e-5, atol=2e-6)
    assert candidate.conv2d.weight.is_contiguous(memory_format=torch.channels_last)
    assert candidate.conv3d.weight.is_contiguous(memory_format=torch.channels_last_3d)


def test_invalid_and_meta_layout_requests_fail_before_mutation():
    model = _MixedDecoder()
    original = model.conv2d.weight.data_ptr()
    model.conv3d.to("meta")
    with pytest.raises(RuntimeError, match="meta"):
        convert_convolution_weight_layouts(model, conv2d=True, conv3d=True)
    assert model.conv2d.weight.data_ptr() == original
    with pytest.raises(TypeError):
        convert_convolution_weight_layouts(model, conv2d="yes")
    with pytest.raises(ValueError, match="no matching"):
        convert_convolution_weight_layouts(torch.nn.Linear(2, 3), conv2d=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mixed_layout_inductor_preserves_decode_pixels():
    torch.manual_seed(81)
    reference = _MixedDecoder().eval().to("cuda")
    candidate = copy.deepcopy(reference)
    convert_convolution_weight_layouts(candidate, conv2d=True, conv3d=True)
    optimized = torch.compile(candidate, backend="inductor", fullgraph=True)
    with torch.no_grad():
        for index in range(2):
            value = torch.randn(1, 4, 2, 5, 7, device="cuda") + index
            torch.testing.assert_close(optimized(value), reference(value), rtol=2e-5, atol=5e-6)
