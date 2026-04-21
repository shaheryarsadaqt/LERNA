import pytest
from lerna.callbacks.lerna_switching import LERNASwitchingCallback


def test_callback_refuses_momentum_mode():
    with pytest.raises(RuntimeError, match="double-step"):
        LERNASwitchingCallback(ler_tracker=object(), apply_momentum=True)