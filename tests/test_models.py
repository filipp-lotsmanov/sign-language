"""
Basic tests for sign language models and detection pipeline.

Run with: python -m pytest tests/ -v
"""
import pytest
import torch
import numpy as np

from src.backend.models.cnn_model import ResidualMLP, ResidualBlock, ASLClassifier
from src.backend.models.lstm_model import DynamicSignLSTM


class TestResidualBlock:
    """Tests for the ResidualBlock component."""

    def test_output_shape_matches_input(self) -> None:
        block = ResidualBlock(dim=128)
        x = torch.randn(4, 128)
        out = block(x)
        assert out.shape == x.shape

    def test_skip_connection_preserves_gradient(self) -> None:
        block = ResidualBlock(dim=64)
        x = torch.randn(2, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestResidualMLP:
    """Tests for the static sign classifier."""

    def test_default_output_shape(self) -> None:
        model = ResidualMLP()
        x = torch.randn(8, 63)
        out = model(x)
        assert out.shape == (8, 24)

    def test_custom_config(self) -> None:
        model = ResidualMLP(input_dim=42, num_classes=10, hidden_dim=128, num_blocks=2)
        x = torch.randn(4, 42)
        out = model(x)
        assert out.shape == (4, 10)

    def test_single_sample(self) -> None:
        model = ResidualMLP()
        model.eval()
        x = torch.randn(1, 63)
        out = model(x)
        assert out.shape == (1, 24)

    def test_eval_mode_deterministic(self) -> None:
        model = ResidualMLP()
        model.eval()
        x = torch.randn(1, 63)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_softmax_sums_to_one(self) -> None:
        model = ResidualMLP()
        model.eval()
        x = torch.randn(1, 63)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestDynamicSignLSTM:
    """Tests for the dynamic sign classifier."""

    def test_default_output_shape(self) -> None:
        model = DynamicSignLSTM()
        x = torch.randn(4, 30, 63)
        out = model(x)
        assert out.shape == (4, 2)

    def test_custom_config(self) -> None:
        model = DynamicSignLSTM(input_size=42, hidden_size=64, num_classes=3)
        x = torch.randn(2, 30, 42)
        out = model(x)
        assert out.shape == (2, 3)

    def test_single_sample(self) -> None:
        model = DynamicSignLSTM()
        x = torch.randn(1, 30, 63)
        out = model(x)
        assert out.shape == (1, 2)

    def test_variable_sequence_length(self) -> None:
        model = DynamicSignLSTM()
        for seq_len in [10, 30, 50]:
            x = torch.randn(1, seq_len, 63)
            out = model(x)
            assert out.shape == (1, 2)

    def test_eval_mode_deterministic(self) -> None:
        model = DynamicSignLSTM()
        model.eval()
        x = torch.randn(1, 30, 63)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


class TestASLClassifierCompat:
    """Tests for backward compatibility alias."""

    def test_legacy_alias_works(self) -> None:
        model = ASLClassifier(input_size=63, num_classes=24)
        model.eval()
        x = torch.randn(1, 63)
        out = model(x)
        assert out.shape == (1, 24)

    def test_is_residual_mlp(self) -> None:
        model = ASLClassifier(input_size=63, num_classes=24)
        assert isinstance(model, ResidualMLP)
