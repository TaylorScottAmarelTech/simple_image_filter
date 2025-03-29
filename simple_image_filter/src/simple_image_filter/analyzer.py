"""
Image analyzer module for detecting unrealistic images based on RGB histogram analysis.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from .histogram import analyze_histogram, find_histogram_peaks

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """
    Analyzer for detecting unrealistic images based on RGB histogram patterns.
    
    This class analyzes RGB histograms in images to detect the characteristic
    3-spike pattern with low standard deviation that often indicates artificially
    generated or problematic images.
    """
    
    def __init__(
        self, 
        peak_threshold: float = 0.1, 
        max_peaks: int = 3, 
        std_dev_threshold: float = 15.0,
        min_peak_height_ratio: float = 0.2
    ):
        """
        Initialize the image analyzer with configurable thresholds.
        
        Args:
            peak_threshold: Minimum height ratio for a peak to be considered significant
            max_peaks: Maximum number of peaks to look for (default 3)
            std_dev_threshold: Maximum standard deviation for pixel intensities to flag as unrealistic
            min_peak_height_ratio: Minimum ratio of peak height to mean height to be considered a spike
        """
        self.peak_threshold = peak_threshold
        self.max_peaks = max_peaks
        self.std_dev_threshold = std_dev_threshold
        self.min_peak_height_ratio = min_peak_height_ratio
        logger.info(f"Initialized ImageAnalyzer with peak_threshold={peak_threshold}, "
                   f"max_peaks={max_peaks}, std_dev_threshold={std_dev_threshold}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from a file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: The loaded image in BGR format
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be loaded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return img
    
    def analyze_image(self, image: Union[str, np.ndarray]) -> Dict:
        """
        Analyze an image to determine if it has unrealistic histogram characteristics.
        
        Args:
            image: Either a path to an image file or a numpy array containing the image
            
        Returns:
            Dict containing analysis results:
                - is_realistic: Boolean indicating if the image has realistic histograms
                - reason: String explaining why the image was flagged (if applicable)
                - channel_stats: Dictionary with statistics for each color channel
        """
        # Load the image if a path was provided
        if isinstance(image, str):
            img = self.load_image(image)
        else:
            img = image.copy()
        
        # Ensure the image is in BGR format (OpenCV default)
        if len(img.shape) < 3 or img.shape[2] != 3:
            if len(img.shape) == 2:  # Grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError("Image must have 3 channels (BGR/RGB)")
        
        # Analyze the histograms for each channel
        channels = ["blue", "green", "red"]
        channel_stats = {}
        unrealistic_channels = []
        
        for i, channel_name in enumerate(channels):
            # Extract the channel
            channel = img[:, :, i]
            
            # Calculate histogram
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            # Find peaks in the histogram
            peaks, properties = find_histogram_peaks(
                hist, 
                height=self.peak_threshold,
                distance=10,
                prominence=0.05
            )
            
            # Calculate standard deviation
            std_dev = np.std(channel)
            
            # Analyze histogram shape
            histogram_analysis = analyze_histogram(
                hist, 
                peaks, 
                self.max_peaks, 
                self.min_peak_height_ratio
            )
            
            # Store stats for this channel
            channel_stats[channel_name] = {
                "std_dev": float(std_dev),
                "peak_count": len(peaks),
                "peaks": peaks.tolist(),
                "histogram": hist.tolist(),
                "has_spike_pattern": histogram_analysis["has_spike_pattern"]
            }
            
            # Check if this channel has unrealistic characteristics
            if (histogram_analysis["has_spike_pattern"] and std_dev < self.std_dev_threshold):
                unrealistic_channels.append(channel_name)
        
        # Determine if the image is realistic based on channel analysis
        is_realistic = len(unrealistic_channels) < 2  # Flag if 2+ channels are unrealistic
        
        reason = ""
        if not is_realistic:
            reason = f"Detected unrealistic histogram patterns in {', '.join(unrealistic_channels)} channels"
            if len(unrealistic_channels) == 3:
                reason += " with characteristic 3-spike pattern and low standard deviation"
        
        logger.info(f"Image analyzed: realistic={is_realistic}")
        if not is_realistic:
            logger.info(f"Rejection reason: {reason}")
        
        return {
            "is_realistic": is_realistic,
            "reason": reason,
            "channel_stats": channel_stats
        }
    
    def is_image_realistic(self, image: Union[str, np.ndarray]) -> bool:
        """
        Quick check if an image has realistic histogram characteristics.
        
        Args:
            image: Either a path to an image file or a numpy array containing the image
            
        Returns:
            Boolean indicating if the image has realistic histograms
        """
        analysis = self.analyze_image(image)
        return analysis["is_realistic"]
