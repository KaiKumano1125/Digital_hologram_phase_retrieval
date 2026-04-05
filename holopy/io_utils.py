"""
Image I/O utilities.

These are the only functions in holopy/ that touch the filesystem.
They sit here rather than in pipeline/ because they are used by both
the simulator and the reconstruction loop.
"""

import cv2
import numpy as np
import torch


def read_image(filename: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Read a grayscale image and return a float32 tensor normalised to [0, 1].

    Args:
        filename: Path to the image file.
        device: Torch device for the returned tensor.

    Returns:
        Normalised image tensor, shape (H, W), dtype float32.

    Raises:
        FileNotFoundError: If the image cannot be opened.
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    normalised = img.astype(np.float32) / 255.0
    return torch.tensor(normalised, dtype=torch.float32, device=device)


def save_intensity(intensity_tensor: torch.Tensor, filename: str) -> None:
    """Peak-normalise a real tensor and save it as an 8-bit grayscale PNG.

    Args:
        intensity_tensor: 2-D real tensor of any dtype. Detached from grad graph.
        filename: Output file path (parent directory must exist).
    """
    arr = intensity_tensor.detach().cpu().float().numpy()
    max_val = float(np.max(arr))
    if max_val > 0:
        arr = (arr / max_val) * 255.0
    else:
        arr = np.zeros_like(arr)
    cv2.imwrite(filename, arr.astype(np.uint8))
