"""Portable PyTorch device selection."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if not available[requested]:
        raise RuntimeError(f"Requested device '{requested}' is unavailable; use cpu or auto")
    return torch.device(requested)
