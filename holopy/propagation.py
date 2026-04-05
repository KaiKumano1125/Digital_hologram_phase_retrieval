"""
Optical wave propagation kernels.

All functions are pure: they take tensors (and scalar parameters) and return
tensors. No file I/O, no print statements, no global state.
"""

import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift


def angular_function(
    width: int,
    height: int,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the ASM transfer function H(fx, fy) for propagation distance z.

    Computed internally in float64 for numerical stability near the evanescent
    boundary, then cast to complex64 on return.

    # NOTE: precision — float64 is required here to avoid catastrophic
    # cancellation in sqrt((1/λ)² - fx² - fy²) near the evanescent cutoff.
    # Casting to complex64 happens only at the output boundary.

    Args:
        width: Grid width in pixels (x dimension).
        height: Grid height in pixels (y dimension).
        wavelength: Light wavelength in metres.
        z: Propagation distance in metres (negative = back-propagation).
        dx: Pixel pitch in x direction (metres).
        dy: Pixel pitch in y direction (metres).
        device: Torch device for output tensor.

    Returns:
        Transfer function H of shape (height, width), dtype complex64.
    """
    # Frequency axes — kept in float64 for precision
    fy = torch.fft.fftfreq(height, d=dy, device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(width, d=dx, device=device, dtype=torch.float64)
    fy, fx = torch.meshgrid(fy, fx, indexing='ij')

    prop_mask = ((wavelength * fx) ** 2 + (wavelength * fy) ** 2 <= 1.0)

    # kz computed in complex128 to handle the sqrt safely
    kz = torch.sqrt((1.0 / wavelength) ** 2 - fx ** 2 - fy ** 2 + 0j)
    H = torch.exp(1j * 2 * np.pi * kz * z)
    H = H * prop_mask  # zero evanescent components

    return H.to(torch.complex64)


def angular_spectrum_prop(
    input_wave: torch.Tensor,
    transfer_function: torch.Tensor,
    cropped_width: int,
    cropped_height: int,
) -> torch.Tensor:
    """Propagate a wave using a precomputed ASM transfer function, then center-crop.

    Args:
        input_wave: Input complex field, shape (H_pad, W_pad).
        transfer_function: Precomputed H from angular_function(), same spatial shape.
        cropped_width: Output width after center-crop.
        cropped_height: Output height after center-crop.

    Returns:
        Propagated and cropped field, shape (cropped_height, cropped_width).
    """
    propagated = ifft2(fft2(input_wave) * transfer_function)
    start_y = input_wave.shape[0] // 2 - cropped_height // 2
    start_x = input_wave.shape[1] // 2 - cropped_width // 2
    return propagated[start_y:start_y + cropped_height, start_x:start_x + cropped_width]


def create_fresnel_impulse_response(
    width: int,
    height: int,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Fresnel quadratic-phase impulse response h(x,y) = exp(i·k/(2z)·(x²+y²)).

    Args:
        width: Grid width in pixels.
        height: Grid height in pixels.
        wavelength: Light wavelength in metres.
        z: Propagation distance in metres.
        dx: Pixel pitch in x direction (metres).
        dy: Pixel pitch in y direction (metres).
        device: Torch device.

    Returns:
        FFT-shifted impulse response, shape (height, width), dtype complex64.
    """
    k = 2 * np.pi / wavelength
    cx, cy = width / 2.0, height / 2.0
    x = (torch.arange(width, device=device, dtype=torch.float32) - cx) * dx
    y = (torch.arange(height, device=device, dtype=torch.float32) - cy) * dy
    x, y = torch.meshgrid(x, y, indexing='ij')
    r_sq = x ** 2 + y ** 2
    # NOTE: precision — k/(2z)*r_sq can be large; float32 is sufficient here
    # because the Fresnel approximation itself introduces larger error than f32 rounding.
    phase = (k / (2 * z)) * r_sq
    h = torch.exp(1j * phase.to(torch.complex64))
    return fftshift(h)


def fresnel_convolution_prop(
    input_wave: torch.Tensor,
    impulse_response_fft: torch.Tensor,
    crop_width: int,
    crop_height: int,
) -> torch.Tensor:
    """Propagate using a precomputed FFT of the Fresnel impulse response, then center-crop.

    Args:
        input_wave: Input complex field, shape (H_pad, W_pad).
        impulse_response_fft: fft2(create_fresnel_impulse_response(...)), same spatial shape.
        crop_width: Output width after center-crop.
        crop_height: Output height after center-crop.

    Returns:
        Propagated and cropped field, shape (crop_height, crop_width).
    """
    padded_height, padded_width = input_wave.shape
    propagated = ifft2(fft2(input_wave) * impulse_response_fft)
    start_y = padded_height // 2 - crop_height // 2
    start_x = padded_width // 2 - crop_width // 2
    return propagated[start_y:start_y + crop_height, start_x:start_x + crop_width]
