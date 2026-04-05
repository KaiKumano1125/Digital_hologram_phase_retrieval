"""
Loss functions for holographic phase retrieval.

Pure functions: tensors in, scalar tensor out. No side effects.
"""

import torch
import torch.nn.functional as F


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation: sum of absolute horizontal and vertical differences.

    Encourages spatial smoothness in amplitude or phase maps.

    Args:
        img: 2-D real tensor, shape (H, W).

    Returns:
        Scalar TV loss value.
    """
    tv_h = torch.sum(torch.abs(img[:-1, :] - img[1:, :]))
    tv_w = torch.sum(torch.abs(img[:, :-1] - img[:, 1:]))
    return tv_h + tv_w


def mse_loss(
    simulated_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
) -> torch.Tensor:
    """Normalised MSE between simulated and target hologram intensities.

    Both fields are peak-normalised to [0, 1] before comparison so that
    absolute intensity scale does not affect the loss.

    Args:
        simulated_intensity: Simulated |U|², shape (H, W), real.
        target_intensity: Ground-truth hologram intensity, same shape.

    Returns:
        Scalar MSE loss.
    """
    sim_norm = simulated_intensity / (simulated_intensity.max() + 1e-9)
    tgt_norm = target_intensity / (target_intensity.max() + 1e-9)
    return F.mse_loss(sim_norm, tgt_norm)


def reference_intensity_loss(
    R_int: torch.Tensor,
    I_ref: torch.Tensor,
) -> torch.Tensor:
    """Log-domain MSE between simulated and measured reference intensity.

    Using log1p compresses the dynamic range and reduces sensitivity to
    bright-spot outliers compared to linear MSE.

    Args:
        R_int: Simulated reference intensity |R|², shape (H, W), real.
        I_ref: Measured reference intensity (possibly noisy), same shape.

    Returns:
        Scalar reference loss.
    """
    return torch.mean(
        (torch.log1p(R_int + 1e-9) - torch.log1p(I_ref + 1e-9)) ** 2
    )
