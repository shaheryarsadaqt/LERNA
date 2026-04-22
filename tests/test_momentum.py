import torch
from lerna.utils.momentum import apply_momentum_extrapolation


def _toy_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))


def test_adam_bias_corrected_step_moves_in_minus_exp_avg():
    """Test Adam first-moment-only (no v_t scaling) - old behavior."""
    m = _toy_model()
    opt = torch.optim.Adam(m.parameters(), lr=0.1, betas=(0.9, 0.999))
    x = torch.randn(8, 4); y = torch.randint(0, 2, (8,))
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward(); opt.step()
    before = [p.detach().clone() for p in m.parameters()]
    # Use first-moment-only (old behavior, no v_t scaling)
    stats = apply_momentum_extrapolation(opt, adam_use_second_moment=False)
    # New function returns dict with stats
    assert stats['adam_updated'] > 0 or stats['skipped'] > 0
    for b, p, pg in zip(before, m.parameters(), [g for g in opt.param_groups for _ in g["params"]]):
        st = opt.state[p]
        step = int(st["step"])
        bc = 1 - 0.9 ** step
        expected = b - 0.1 * st["exp_avg"] / bc
        assert torch.allclose(p.data, expected, atol=1e-6)


def test_adam_with_second_moment():
    """Test Adam with second moment (v_t scaling) - new default behavior."""
    m = _toy_model()
    opt = torch.optim.Adam(m.parameters(), lr=0.1, betas=(0.9, 0.999))
    x = torch.randn(8, 4); y = torch.randint(0, 2, (8,))
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward(); opt.step()
    before = [p.detach().clone() for p in m.parameters()]
    # Use second moment (new default behavior)
    stats = apply_momentum_extrapolation(opt, adam_use_second_moment=True)
    assert stats['adam_updated'] > 0
    # With second moment, the update includes v_t scaling
    for b, p in zip(before, m.parameters()):
        st = opt.state[p]
        step = int(st["step"])
        bc1 = 1 - 0.9 ** step
        bc2 = 1 - 0.999 ** step
        m_hat = st["exp_avg"] / bc1
        v_hat = st["exp_avg_sq"] / bc2
        denom = (v_hat.sqrt() + 1e-8)
        expected = b - 0.1 * m_hat / denom
        assert torch.allclose(p.data, expected, atol=1e-6)


def test_sgd_uses_momentum_buffer():
    m = _toy_model()
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(8, 4); y = torch.randint(0, 2, (8,))
    opt.zero_grad(); torch.nn.functional.cross_entropy(m(x), y).backward(); opt.step()
    before = [p.detach().clone() for p in m.parameters()]
    stats = apply_momentum_extrapolation(opt)
    # New function returns dict with stats
    assert stats['sgd_updated'] > 0 or stats['skipped'] > 0
    for b, p in zip(before, m.parameters()):
        mb = opt.state[p]["momentum_buffer"]
        assert torch.allclose(p.data, b - 0.1 * mb, atol=1e-7)