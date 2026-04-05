"""
Tests for holopy.wavefield and a full forward-simulation smoke test.

Uses tiny synthetic data (8×8 pixels) so no real image files are needed.
"""

import pytest
import torch

from holopy.wavefield import generate_spherical_reference_wave, generate_two_point_sources
from holopy.propagation import angular_function, angular_spectrum_prop

DEVICE = torch.device("cpu")
WAVELENGTH = 500e-9
Z1 = 0.005
Z2 = 0.0002
DX = 5e-7


class TestSphericalReferenceWave:
    def test_output_shape(self):
        wave = generate_spherical_reference_wave(32, 32, WAVELENGTH, Z1, DEVICE)
        assert wave.shape == (32, 32)

    def test_output_dtype(self):
        wave = generate_spherical_reference_wave(32, 32, WAVELENGTH, Z1, DEVICE)
        assert wave.dtype == torch.complex64

    def test_no_nan_or_inf(self):
        wave = generate_spherical_reference_wave(64, 64, WAVELENGTH, Z1, DEVICE)
        assert not torch.isnan(wave).any()
        assert not torch.isinf(wave).any()

    def test_nonzero_amplitude(self):
        wave = generate_spherical_reference_wave(16, 16, WAVELENGTH, Z1, DEVICE)
        assert wave.abs().max().item() > 0.0


class TestTwoPointSources:
    def test_output_shape(self):
        wave = generate_two_point_sources(32, 32, WAVELENGTH, Z1, offset=5, device=DEVICE)
        assert wave.shape == (32, 32)

    def test_output_dtype(self):
        wave = generate_two_point_sources(32, 32, WAVELENGTH, Z1, offset=5, device=DEVICE)
        assert wave.dtype == torch.complex64

    def test_no_nan_or_inf(self):
        wave = generate_two_point_sources(64, 64, WAVELENGTH, Z1, offset=10, device=DEVICE)
        assert not torch.isnan(wave).any()
        assert not torch.isinf(wave).any()

    def test_differs_from_single_point(self):
        single = generate_spherical_reference_wave(32, 32, WAVELENGTH, Z1, DEVICE)
        two = generate_two_point_sources(32, 32, WAVELENGTH, Z1, offset=5, device=DEVICE)
        assert not torch.allclose(single.abs(), two.abs())


class TestForwardSimulationSmoke:
    """End-to-end forward pass with tiny synthetic data.

    Verifies that the pipeline composes without errors and produces
    non-negative hologram intensities of the correct shape.
    """

    def test_asm_forward_pass(self):
        N, PN = 8, 16
        sy, sx = (PN - N) // 2, (PN - N) // 2

        # Synthetic object: uniform amplitude, zero phase
        amp = torch.ones(N, N, dtype=torch.float32)
        phase = torch.zeros(N, N, dtype=torch.float32)
        object_wave = torch.polar(amp, phase)

        padded = torch.zeros(PN, PN, dtype=torch.complex64)
        padded[sy:sy + N, sx:sx + N] = object_wave

        H_z2 = angular_function(PN, PN, WAVELENGTH, Z2, DX, DX, DEVICE)
        H_ref = angular_function(PN, PN, WAVELENGTH, Z1 + Z2, DX, DX, DEVICE)

        ref_src = generate_spherical_reference_wave(PN, PN, WAVELENGTH, Z1 + Z2, DEVICE)
        ref_at_holo = angular_spectrum_prop(ref_src, H_ref, N, N)
        obj_at_holo = angular_spectrum_prop(padded, H_z2, N, N)

        total = ref_at_holo + obj_at_holo
        intensity = torch.abs(total) ** 2

        assert intensity.shape == (N, N)
        assert (intensity >= 0).all()
        assert not torch.isnan(intensity).any()
        assert intensity.max().item() > 0.0
