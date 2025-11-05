import pytest
import torch

from squareq.quant.modules_lora import QuantLinearLoRA


@pytest.mark.cuda
def test_lora_receives_grads():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    layer = QuantLinearLoRA(4, 3, rank=2, alpha=1.0)
    layer.set_quant_state(
        qweight=torch.randint(-128, 127, (3, 4), dtype=torch.int8),
        scale=torch.ones(3, dtype=torch.float32),
        zero_point=torch.zeros(3, dtype=torch.float32),
        bias=torch.zeros(3, dtype=torch.float32),
    )
    layer = layer.to("cuda")
    x = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16, requires_grad=False)
    target = torch.zeros(2, 3, device="cuda", dtype=torch.bfloat16)
    out = layer(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert layer.qweight.grad is None
