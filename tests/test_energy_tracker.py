from lerna.callbacks.lerna_switching import EnergyTracker


def test_estimate_energy_saved_is_deprecated_and_returns_zero():
    t = EnergyTracker(gpu_id=0)  # survives without NVML for this test
    assert t.estimate_energy_saved(skipped_backward=True) == 0.0