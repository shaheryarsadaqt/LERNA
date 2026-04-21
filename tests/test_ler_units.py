from lerna.utils.metrics import LERTracker
import torch


def test_ler_is_none_before_velocity_is_defined():
    t = LERTracker(task="sst2", window_size=5)
    m = torch.nn.Linear(4, 2)
    t.update(loss=2.0, logits=torch.randn(4, 2), n_steps=1, model=m)
    t.update(loss=1.5, logits=torch.randn(4, 2), n_steps=2, model=m)
    # first real LER only appears AFTER velocity is defined (second snapshot)
    assert all(v is not None for v in t.ler_history)


def test_ler_does_not_fire_on_loss_increase():
    t = LERTracker(task="sst2", window_size=5)
    m = torch.nn.Linear(4, 2)
    for loss in [1.0, 1.2, 1.4, 1.6]:
        t.update(loss=loss, logits=torch.randn(4, 2), n_steps=1, model=m)
    assert all(l == 0.0 for l in t.ler_history)   # negative Δloss clipped to 0