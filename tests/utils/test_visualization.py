"""Unit tests for visualization utilities."""

import numpy as np
import torch

from jute_disease.utils.visualization import denormalize


def test_denormalize():
    """Test denormalize function converts normalized tensor back to numpy image."""
    # (C, H, W) tensor
    img_tensor = torch.zeros((3, 32, 32))

    # Use default ImageNet mean/std
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    denorm = denormalize(img_tensor, mean=mean, std=std)

    assert isinstance(denorm, np.ndarray)
    assert denorm.shape == (32, 32, 3)

    # Normalized 0 should become mean
    expected_pixel = np.array(mean)
    assert np.allclose(denorm[0, 0, :], expected_pixel, atol=1e-5)


def test_denormalize_clipping():
    """Test denormalize function clips values to [0, 1]."""
    # Large positive/negative values
    img_tensor = torch.ones((3, 10, 10)) * 10
    denorm_high = denormalize(img_tensor)
    assert np.all(denorm_high <= 1.0)

    img_tensor_low = torch.ones((3, 10, 10)) * -10
    denorm_low = denormalize(img_tensor_low)
    assert np.all(denorm_low >= 0.0)
