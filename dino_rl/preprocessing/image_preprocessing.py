"""Shared RGB-to-grayscale pipeline for simulator frames and future screenshots."""

import numpy as np
from PIL import Image

from dino_rl.config import ImageConfig


class ImagePreprocessor:
    """Crop, convert to luminance and bilinearly resize; optionally normalize.

    Inputs must be HWC uint8 RGB. BGR/BGRA captures require an explicit adapter,
    preventing silent color/channel mistakes when Chrome capture is added later.
    No game state is read here. Output shape is (height, width).
    """

    def __init__(self, config: ImageConfig | None = None):
        self.config = config or ImageConfig()

    def crop_grayscale(self, rgb: np.ndarray) -> np.ndarray:
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB numpy array with dtype uint8")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
            raise ValueError("Expected nonempty RGB pixels with shape (height, width, 3)")
        image = Image.fromarray(rgb)
        if self.config.crop is not None:
            left, top, right, bottom = self.config.crop
            if right > rgb.shape[1] or bottom > rgb.shape[0]:
                raise ValueError(
                    f"Crop {self.config.crop} exceeds RGB frame dimensions {rgb.shape[:2]}; adjust image.crop"
                )
            image = image.crop((left, top, right, bottom))
        return np.array(image.convert("L"), dtype=np.uint8)

    def process_uint8(self, rgb: np.ndarray) -> np.ndarray:
        """Return compact grayscale pixels in [0, 255] for stacking/storage."""
        grayscale = self.crop_grayscale(rgb)
        resized = Image.fromarray(grayscale).resize(
            (self.config.width, self.config.height), Image.Resampling.BILINEAR
        )
        return np.array(resized, dtype=np.uint8)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Return a model-ready float32 frame in [0, 1]."""
        return self.process_uint8(rgb).astype(np.float32) / np.float32(255.0)
