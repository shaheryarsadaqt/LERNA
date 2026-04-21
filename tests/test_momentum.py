import torch
from lerna.utils.momentum import apply_momentum_extrapolation


def _toy_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))


def test_adam_bias_corrected_step_moves_in_minus_exp_avg():
    m = _toy_model()
    opt = torch.optim.Adam(m.parameters(), lr=0.1, betas=(0.9, 0.999))
    x = torch.randn(8, 4); y = torch.randint(0, 2, (8,))
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward(); opt.step()
    before = [p.detach().clone() for p in m.parameters()]
    n = apply_momentum_extrapolation(opt)
    assert n > 0
    for b, p, pg in zip(before, m.parameters(), [g for g in opt.param_groups for _ in g["params"]]):
        st = opt.state[p]
        step = int(st["step"])
        bc = 1 - 0.9 ** step
        expected = b - 0.1 * st["exp_avg"] / bc
        assert torch.allclose(p.data, expected, atol=1e-6)


def test_sgd_uses_momentum_buffer():
    m = _toy_model()
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(8, 4); y = torch.randint(0, 2, (8,))
    opt.zero_grad(); torch.nn.functional.cross_entropy(m(x), y).backward(); opt.step()
    before = [p.detach().clone() for p in m.parameters()]
    apply_momentum_extrapolation(opt)
    for b, p in zip(before, m.parameters()):
        mb = opt.state[p]["momentum_buffer"]
        assert torch.allclose(p.data, b - 0.1 * mb, atol=1e-7)