"""
Tests for the is_image_ok function in the analyzer module.
"""
import numpy as np
import pytest
from PIL import Image

from simple_image_filter.analyzer import is_image_ok


def test_is_image_ok_perfect_image():
    """Test that a perfect image passes the check."""
    # Create a synthetic image with good properties
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Medium gray
    
    # Add some variation for contrast
    image[25:75, 25:75] = 200  # Lighter rectangle in the middle
    
    result, details = is_image_ok(image)
    assert result is True
    
    
def test_is_image_ok_too_dark():
    """Test that a too dark image fails the check."""
    # Create a very dark image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 20  # Very dark gray
    
    result, details = is_image_ok(image)
    assert result is False
    assert details['is_brightness_ok'] is False
    
    
def test_is_image_ok_too_bright():
    """Test that a too bright image fails the check."""
    # Create a very bright image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 240  # Very bright
    
    result, details = is_image_ok(image)
    assert result is False
    assert details['is_brightness_ok'] is False
    
    
def test_is_image_ok_no_contrast():
    """Test that an image with no contrast fails the check."""
    # Create an image with no contrast (solid color)
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Medium gray everywhere
    
    result, details = is_image_ok(image)
    assert result is False
    assert details['is_contrast_ok'] is False
    
    
def test_is_image_ok_custom_thresholds():
    """Test using custom thresholds."""
    # Create a medium brightness image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 150
    
    # With default thresholds it should pass
    result1, _ = is_image_ok(image)
    assert result1 is True
    
    # With stricter thresholds it should fail
    custom_thresholds = {
        'brightness_max': 140  # Lower than the image's brightness
    }
    result2, details = is_image_ok(image, custom_thresholds)
    assert result2 is False
    assert details['is_brightness_ok'] is False


def test_is_image_ok_with_pil_image():
    """Test that the function accepts PIL Image objects."""
    # Create a PIL Image
    pil_image = Image.new('RGB', (100, 100), color='gray')
    
    # Should not raise any exceptions
    result, _ = is_image_ok(pil_image)
    assert isinstance(result, bool)