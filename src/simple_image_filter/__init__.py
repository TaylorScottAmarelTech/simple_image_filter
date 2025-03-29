"""
Simple Image Filter - A package for analyzing and filtering images.
"""

from .analyzer import (
    calculate_brightness,
    calculate_contrast,
    calculate_saturation,
    calculate_sharpness,
    is_image_ok  # Expose the new function
)

__all__ = [
    'calculate_brightness',
    'calculate_contrast',
    'calculate_saturation',
    'calculate_sharpness',
    'is_image_ok'
]