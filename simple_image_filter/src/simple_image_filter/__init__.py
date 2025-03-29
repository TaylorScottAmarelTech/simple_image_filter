"""
Simple Image Filter
=====================

A package for filtering unrealistic images based on RGB histogram analysis.
Detects images with abnormal histograms showing 3 distinct spikes with 
small standard deviation.
"""

__version__ = "0.1.0"

from .analyzer import ImageAnalyzer
from .histogram import analyze_histogram, find_histogram_peaks

__all__ = [
    'ImageAnalyzer',
    'analyze_histogram',
    'find_histogram_peaks',
]
