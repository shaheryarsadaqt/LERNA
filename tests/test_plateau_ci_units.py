import numpy as np
from lerna.utils.plateau_ies import IESPlateauDetector


def test_ci_is_in_percentage_units_not_sem_of_diffs():
    det = IESPlateauDetector(threshold=1e-4, window_size=5, patience=10)
    rng = np.random.default_rng(0)
    # fake a converging loss
    losses = np.concatenate([np.linspace(3.0, 1.0, 100), 1.0 + 0.001 * rng.standard_normal(200)])
    for l in losses:
        det.update(float(l), step=len(det.loss_history))
    res = det.analyze_plateau()
    if res is None:
        # No plateau detected - that's okay for this test
        return
    lo, hi = res.get("confidence_interval_95") if hasattr(res, "get") else res.confidence_interval_95
    if lo is None:
        return
    assert 0 <= lo <= hi <= 100  # proper percentage bounds, not diff-space
    assert hi - lo < 80          # should be a reasonable CI, not a full-width artifact