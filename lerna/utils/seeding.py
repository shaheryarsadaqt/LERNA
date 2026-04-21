"""Canonical seeding. Use this EVERYWHERE; do not reimplement in scripts."""
import os, random, numpy as np, torch

def set_global_seed(seed: int, deterministic: bool = True,
                    warn_only: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

def load_determinism_from_yaml(cfg: dict) -> None:
    det = bool(cfg.get("cudnn_deterministic", True))
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed, deterministic=det, warn_only=cfg.get("warn_only", False))