"""Validation helpers for collated ML batch fields."""

from torch import Tensor


# ADD 2026-08-19: Return a required tensor field or fail before model execution.
def require_batch_tensor(batch: dict[object, object], field: str) -> Tensor:
    """Return a required tensor field or fail before model execution."""
    value = batch.get(field)
    if not isinstance(value, Tensor):
        raise TypeError(f"Batch field '{field}' must be a torch.Tensor.")
    return value
