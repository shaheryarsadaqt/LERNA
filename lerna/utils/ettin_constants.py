ETTIN_MODEL_ID = "jhu-clsp/ettin-encoder-150m"
ETTIN_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"
SYNTHETIC_MODEL_ID = "tiny-linear-cpu"
SYNTHETIC_MODEL_REVISION = "synthetic-cpu-local-v1"


def validate_ettin_revision(model_name, model_revision):
    """Validate Ettin revision without external dependencies."""
    if model_name != ETTIN_MODEL_ID:
        return None
    if not model_revision:
        raise ValueError(
            f"Ettin model requires an explicit revision; "
            f"expected {ETTIN_REVISION!r}"
        )
    if not isinstance(model_revision, str):
        raise ValueError(
            "Ettin revision must be a 40-character lowercase hex SHA; "
            f"got {model_revision!r}"
        )
    if model_revision != model_revision.lower():
        raise ValueError(
            f"Ettin revision must be lowercase SHA; got {model_revision!r}"
        )
    if len(model_revision) != 40 or any(
        ch not in "0123456789abcdef" for ch in model_revision
    ):
        raise ValueError(
            f"Ettin revision must be a 40-character lowercase hex SHA; "
            f"got {model_revision!r}"
        )
    if model_revision != ETTIN_REVISION:
        raise ValueError(
            f"Ettin revision mismatch: expected {ETTIN_REVISION!r}, "
            f"got {model_revision!r}"
        )
    return model_revision
