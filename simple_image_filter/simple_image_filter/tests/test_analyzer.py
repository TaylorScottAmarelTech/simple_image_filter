"""
Unit tests for the image analyzer.
"""

import unittest
import numpy as np
import cv2
from simple_image_filter.analyzer import ImageAnalyzer
from simple_image_filter.histogram import find_histogram_peaks, analyze_histogram

class TestHistogramAnalysis(unittest.TestCase):
    """Test cases for histogram analysis functions."""
    
    def test_find_histogram_peaks(self):
        """Test peak detection in histograms."""
        # Create a test histogram with three clear peaks
        test_hist = np.zeros(256)
        test_hist[50] = 0.2  # Peak 1
        test_hist[125] = 0.3  # Peak 2
        test_hist[200] = 0.25  # Peak 3
        # Add some noise
        test_hist += np.random.rand(256) * 0.01
        
        # Find peaks
        peaks, _ = find_histogram_peaks(test_hist, height=0.1, distance=10)
        
        # Should identify exactly 3 peaks
        self.assertEqual(len(peaks), 3)
        self.assertTrue(50 in peaks)
        self.assertTrue(125 in peaks)
        self.assertTrue(200 in peaks)
    
    def test_analyze_histogram(self):
        """Test histogram pattern analysis."""
        # Create a test histogram with three clear peaks (typical of unrealistic images)
        test_hist = np.zeros(256)
        test_hist[50] = 0.2  # Peak 1
        test_hist[125] = 0.3  # Peak 2
        test_hist[200] = 0.25  # Peak 3
        # Add some noise
        test_hist += np.random.rand(256) * 0.01
        
        # Find peaks
        peaks, _ = find_histogram_peaks(test_hist, height=0.1, distance=10)
        
        # Analyze histogram
        analysis = analyze_histogram(test_hist, peaks)
        
        # Should detect the 3-spike pattern
        self.assertTrue(analysis["has_spike_pattern"])
        self.assertEqual(analysis["num_peaks"], 3)
        
        # Create a more natural histogram (gradual distribution)
        natural_hist = np.zeros(256)
        for i in range(256):
            natural_hist[i] = 0.1 + 0.3 * np.sin(i / 256 * np.pi)
        natural_hist /= np.sum(natural_hist)  # Normalize
        
        # Find peaks in natural histogram
        natural_peaks, _ = find_histogram_peaks(natural_hist, height=0.1, distance=10)
        
        # Analyze natural histogram
        natural_analysis = analyze_histogram(natural_hist, natural_peaks)
        
        # Should not detect the 3-spike pattern
        self.assertFalse(natural_analysis["has_spike_pattern"])

class TestImageAnalyzer(unittest.TestCase):
    """Test cases for the ImageAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ImageAnalyzer(
            peak_threshold=0.1,
            std_dev_threshold=15.0
        )
        
    def test_analyze_synthetic_images(self):
        """Test analysis on synthetic test images."""
        # Create a synthetic unrealistic image (3 distinct color bands)
        unrealistic_img = np.zeros((300, 300, 3), dtype=np.uint8)
        unrealistic_img[:100, :, 0] = 200  # Blue band
        unrealistic_img[100:200, :, 1] = 200  # Green band
        unrealistic_img[200:, :, 2] = 200  # Red band
        
        # Analyze the unrealistic image
        result = self.analyzer.analyze_image(unrealistic_img)
        
        # Should be classified as unrealistic
        self.assertFalse(result["is_realistic"])
        
        # Create a more natural gradient image
        y, x = np.mgrid[0:300, 0:300]
        natural_img = np.zeros((300, 300, 3), dtype=np.uint8)
        natural_img[:, :, 0] = np.clip(x / 300 * 255, 0, 255).astype(np.uint8)  # Blue gradient
        natural_img[:, :, 1] = np.clip(y / 300 * 255, 0, 255).astype(np.uint8)  # Green gradient
        natural_img[:, :, 2] = np.clip(((x+y) / 600) * 255, 0, 255).astype(np.uint8)  # Red gradient
        
        # Analyze the natural image
        natural_result = self.analyzer.analyze_image(natural_img)
        
        # Should be classified as realistic
        self.assertTrue(natural_result["is_realistic"])

if __name__ == "__main__":
    unittest.main()
