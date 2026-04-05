"""
Reference wave generators.

Pure functions: scalar parameters + device → complex tensor.
"""

import numpy as np
import torch


def generate_spherical_reference_wave(
    width: int,
    height: int,
    wavelength: float,
    z: float,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a spherical (point-source) reference wave at distance z.

    The source is placed at the grid centre, z metres from the sensor plane.
    Amplitude follows 1/r decay; phase follows exp(i·k·r).

    Args:
        width: Grid width in pixels.
        height: Grid height in pixels.
        wavelength: Light wavelength in metres.
        z: Source-to-plane distance in metres.
        device: Torch device.

    Returns:
        Complex wave field, shape (height, width), dtype complex64.
    """
    k = 2 * np.pi / wavelength
    cx, cy = width // 2, height // 2
    x = (torch.arange(width, device=device, dtype=torch.float32) - cx)
    y = (torch.arange(height, device=device, dtype=torch.float32) - cy)
    x, y = torch.meshgrid(x, y, indexing='ij')

    r_sq = x ** 2 + y ** 2 + float(z) ** 2
    r = torch.sqrt(r_sq)
    # Guard against division by zero at the source point
    r = torch.where(r == 0, torch.tensor(1e-9, dtype=torch.float32, device=device), r)

    wave = torch.exp(1j * k * r.to(torch.complex64)) / r.to(torch.complex64)
    return wave.to(torch.complex64)


def generate_two_point_sources(
    width: int,
    height: int,
    wavelength: float,
    z: float,
    offset: int = 100,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a reference wave from two symmetric point sources.

    Sources are placed at (±offset, 0), z metres from the sensor plane.
    The output is the coherent superposition of the two spherical waves.

    Args:
        width: Grid width in pixels.
        height: Grid height in pixels.
        wavelength: Light wavelength in metres.
        z: Source-to-plane distance in metres.
        offset: Lateral pixel offset of each source from the centre.
        device: Torch device.

    Returns:
        Complex wave field, shape (height, width), dtype complex64.
    """
    k = 2 * np.pi / wavelength
    cx, cy = width // 2, height // 2
    x = (torch.arange(width, device=device, dtype=torch.float32) - cx)
    y = (torch.arange(height, device=device, dtype=torch.float32) - cy)
    x, y = torch.meshgrid(x, y, indexing='ij')

    r1 = torch.sqrt((x - offset) ** 2 + y ** 2 + float(z) ** 2).to(torch.complex64)
    r2 = torch.sqrt((x + offset) ** 2 + y ** 2 + float(z) ** 2).to(torch.complex64)

    wave1 = torch.exp(1j * k * r1) / r1
    wave2 = torch.exp(1j * k * r2) / r2
    return (wave1 + wave2).to(torch.complex64)
