import pytest
import torch
from lerna.utils.seeding import set_global_seed


@pytest.fixture(autouse=True)
def _seed_all():
    set_global_seed(0, deterministic=True, warn_only=True)