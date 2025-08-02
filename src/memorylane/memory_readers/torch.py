import torch


def get_memory_usage(device: str | None = None) -> tuple[int, int]:
    """Return current and peak allocated CUDA memory in bytes.

    Returns
    -------
    tuple[float, float]
        ``(current_allocated_bytes, peak_allocated_bytes)``
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
        else:
            raise RuntimeError(
                f"Was given {device=!r}, but {torch.cuda.is_available()=}."
            )
    elif device == "cpu":
        raise NotImplementedError(
            f"Was given {device=!r}, but CPU memory usage tracking is not implemented yet."
        )
        return torch.get_allocated_memory(), torch.get_peak_allocated_memory()
    else:
        raise ValueError(
            f"Invalid device specification: {device=!r}. Expected 'cuda' or 'cpu'."
        )
