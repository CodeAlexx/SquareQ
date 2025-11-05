import pytest
import torch

from squareq.quant.modules import QuantLinear


@pytest.mark.cuda
def test_quant_linear_buffers_remain_int8():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    layer = QuantLinear(4, 3, bias=True)
    qweight = torch.randint(-128, 127, (3, 4), dtype=torch.int8)
    scale = torch.ones(3, dtype=torch.float32)
    zero = torch.zeros(3, dtype=torch.float32)
    bias = torch.zeros(3, dtype=torch.float32)
    layer.set_quant_state(qweight=qweight, scale=scale, zero_point=zero, bias=bias)
    layer = layer.to("cuda")
    assert layer.qweight.dtype == torch.int8
    assert layer.qweight.is_cuda
    assert layer.scale.dtype == torch.float32
    assert layer.zero_point.dtype == torch.float32
    assert layer.bias.dtype == torch.float32
    x = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (2, 3)
