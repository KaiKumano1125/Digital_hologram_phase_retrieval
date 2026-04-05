"""
Tests for holopy.loss — sanity checks on loss function values and shapes.
"""

import pytest
import torch

from holopy.loss import total_variation_loss, mse_loss, reference_intensity_loss


class TestTotalVariationLoss:
    def test_zero_for_uniform_image(self):
        img = torch.ones(32, 32)
        assert total_variation_loss(img).item() == pytest.approx(0.0)

    def test_positive_for_non_uniform(self):
        img = torch.rand(32, 32)
        assert total_variation_loss(img).item() > 0.0

    def test_returns_scalar(self):
        img = torch.rand(16, 16)
        out = total_variation_loss(img)
        assert out.shape == torch.Size([])

    def test_gradient_flows(self):
        img = torch.rand(8, 8, requires_grad=True)
        loss = total_variation_loss(img)
        loss.backward()
        assert img.grad is not None


class TestMseLoss:
    def test_zero_for_identical_inputs(self):
        x = torch.rand(32, 32)
        assert mse_loss(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_different_inputs(self):
        x = torch.rand(32, 32)
        y = torch.rand(32, 32)
        assert mse_loss(x, y).item() > 0.0

    def test_returns_scalar(self):
        x, y = torch.rand(16, 16), torch.rand(16, 16)
        assert mse_loss(x, y).shape == torch.Size([])

    def test_gradient_flows(self):
        sim = torch.rand(8, 8, requires_grad=True)
        tgt = torch.rand(8, 8)
        mse_loss(sim, tgt).backward()
        assert sim.grad is not None


class TestReferenceIntensityLoss:
    def test_zero_for_identical_inputs(self):
        x = torch.rand(32, 32).abs()
        assert reference_intensity_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_inputs(self):
        x = torch.rand(32, 32).abs()
        y = torch.rand(32, 32).abs()
        assert reference_intensity_loss(x, y).item() > 0.0

    def test_returns_scalar(self):
        x, y = torch.rand(16, 16), torch.rand(16, 16)
        assert reference_intensity_loss(x, y).shape == torch.Size([])
