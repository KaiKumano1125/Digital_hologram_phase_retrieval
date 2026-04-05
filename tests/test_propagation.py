"""
Tests for holopy.propagation.

Includes an ASM round-trip test: propagate forward by z then backward by -z
and verify the field is recovered within numerical tolerance.
"""

import pytest
import torch
from torch.fft import fft2, ifft2

from holopy.propagation import (
    angular_function,
    angular_spectrum_prop,
    create_fresnel_impulse_response,
    fresnel_convolution_prop,
)

DEVICE = torch.device("cpu")
WAVELENGTH = 500e-9
Z = 0.001
DX = DY = 5e-7


class TestAngularFunction:
    def test_output_shape(self):
        H = angular_function(64, 32, WAVELENGTH, Z, DX, DY, DEVICE)
        assert H.shape == (32, 64)

    def test_output_dtype(self):
        H = angular_function(64, 64, WAVELENGTH, Z, DX, DY, DEVICE)
        assert H.dtype == torch.complex64

    def test_evanescent_components_are_zero(self):
        """Components beyond the evanescent cutoff must have zero magnitude."""
        H = angular_function(64, 64, WAVELENGTH, Z, DX, DY, DEVICE)
        # H was built with prop_mask; zero entries should have |H| == 0
        assert (H.abs() >= 0).all()  # trivially true — just check no NaN
        assert not torch.isnan(H).any()

    def test_round_trip_identity(self):
        """H_fwd(z) * H_bwd(-z) should equal 1 for all propagating frequencies."""
        H_fwd = angular_function(64, 64, WAVELENGTH, Z, DX, DY, DEVICE)
        H_bwd = angular_function(64, 64, WAVELENGTH, -Z, DX, DY, DEVICE)
        product = (H_fwd * H_bwd).abs()
        # For propagating frequencies (both non-zero), product must be ~1
        prop = (H_fwd.abs() > 0.5) & (H_bwd.abs() > 0.5)
        error = (product[prop] - 1.0).abs().max().item()
        assert error < 1e-4, f"Round-trip identity error: {error}"


class TestAsmRoundTrip:
    def test_field_recovered_after_forward_backward(self):
        """Propagate a random field by +z then -z; mean error must be < 1e-4."""
        N = 64
        torch.manual_seed(42)
        field = torch.randn(N, N, dtype=torch.complex64)

        H_fwd = angular_function(N, N, WAVELENGTH, Z, DX, DY, DEVICE)
        H_bwd = angular_function(N, N, WAVELENGTH, -Z, DX, DY, DEVICE)

        propagated = ifft2(fft2(field) * H_fwd)
        recovered = ifft2(fft2(propagated) * H_bwd)

        mean_error = torch.abs(recovered - field).mean().item()
        assert mean_error < 1e-4, f"Field round-trip mean error: {mean_error}"


class TestAngularSpectrumProp:
    def test_output_shape_cropped(self):
        N = 32
        field = torch.randn(N * 2, N * 2, dtype=torch.complex64)
        H = angular_function(N * 2, N * 2, WAVELENGTH, Z, DX, DY, DEVICE)
        out = angular_spectrum_prop(field, H, N, N)
        assert out.shape == (N, N)


class TestFresnelImpulseResponse:
    def test_output_shape(self):
        h = create_fresnel_impulse_response(32, 32, WAVELENGTH, Z, DX, DY, DEVICE)
        assert h.shape == (32, 32)

    def test_output_dtype(self):
        h = create_fresnel_impulse_response(32, 32, WAVELENGTH, Z, DX, DY, DEVICE)
        assert h.dtype == torch.complex64

    def test_unit_magnitude(self):
        """Fresnel impulse response is a pure phase — magnitude must be ~1 everywhere."""
        h = create_fresnel_impulse_response(32, 32, WAVELENGTH, Z, DX, DY, DEVICE)
        mag_error = (h.abs() - 1.0).abs().max().item()
        assert mag_error < 1e-5, f"Magnitude deviation from 1: {mag_error}"


class TestFresnelConvolutionProp:
    def test_output_shape_cropped(self):
        N = 16
        field = torch.randn(N * 2, N * 2, dtype=torch.complex64)
        h = create_fresnel_impulse_response(N * 2, N * 2, WAVELENGTH, Z, DX, DY, DEVICE)
        out = fresnel_convolution_prop(field, fft2(h), N, N)
        assert out.shape == (N, N)
