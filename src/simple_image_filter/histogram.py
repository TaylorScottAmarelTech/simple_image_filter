"""
Histogram analysis functions for detecting unrealistic patterns in image histograms.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.signal import find_peaks

def find_histogram_peaks(
    histogram: np.ndarray,
    height: float = 0.01,
    distance: int = 5,
    prominence: float = 0.01
) -> Tuple[np.ndarray, Dict]:
    """
    Find peaks in an image histogram.
    
    Args:
        histogram: Normalized histogram array
        height: Minimum height for a peak to be considered
        distance: Minimum horizontal distance between peaks
        prominence: Minimum prominence of peaks
        
    Returns:
        Tuple of (peaks array, peak properties)
    """
    peaks, properties = find_peaks(
        histogram,
        height=height,
        distance=distance,
        prominence=prominence
    )
    
    return peaks, properties

def analyze_histogram(
    histogram: np.ndarray,
    peaks: np.ndarray,
    max_peaks: int = 3,
    min_peak_height_ratio: float = 0.2
) -> Dict:
    """
    Analyze a histogram to detect unrealistic patterns like the 3-spike pattern.
    
    Args:
        histogram: Normalized histogram array
        peaks: Array of peak indices
        max_peaks: Maximum number of peaks to consider (for 3-spike pattern)
        min_peak_height_ratio: Minimum ratio of peak height to mean for a significant spike
        
    Returns:
        Dictionary with analysis results
    """
    # Less than 2 peaks is usually fine (can be gradient, flat, or single dominant color)
    if len(peaks) < 2:
        return {
            "has_spike_pattern": False,
            "num_peaks": len(peaks)
        }
    
    # Sort peaks by height (descending)
    peak_heights = histogram[peaks]
    sorted_indices = np.argsort(-peak_heights)
    top_peaks = peaks[sorted_indices]
    
    # Only consider the top max_peaks
    if len(top_peaks) > max_peaks:
        top_peaks = top_peaks[:max_peaks]
    
    # Calculate average height of significant peaks
    significant_heights = histogram[top_peaks]
    mean_height = np.mean(significant_heights)
    
    # Check if we have exactly 3 significant peaks
    # Artificial images often have exactly 3 spikes (RGB primaries)
    has_three_spike_pattern = False
    
    if len(top_peaks) == 3:
        # Check if all peaks are significantly higher than the rest of the histogram
        avg_non_peak = np.mean(np.delete(histogram, top_peaks))
        peak_to_avg_ratio = mean_height / (avg_non_peak + 1e-10)  # Avoid division by zero
        
        # Check if peaks are roughly evenly spaced
        # This is characteristic of some AI-generated image errors
        peak_distances = np.diff(np.sort(top_peaks))
        peak_distance_std = np.std(peak_distances) if len(peak_distances) > 1 else 0
        
        # Check for the 3-spike pattern
        has_three_spike_pattern = (
            peak_to_avg_ratio > min_peak_height_ratio and
            all(histogram[p] > avg_non_peak * min_peak_height_ratio for p in top_peaks)
        )
    
    return {
        "has_spike_pattern": has_three_spike_pattern,
        "num_peaks": len(peaks),
        "significant_peaks": len(top_peaks),
        "peak_indices": top_peaks.tolist(),
        "peak_heights": [float(histogram[p]) for p in top_peaks]
    }

def calculate_histogram_metrics(histogram: np.ndarray) -> Dict:
    """
    Calculate various metrics for a histogram.
    
    Args:
        histogram: Normalized histogram array
        
    Returns:
        Dictionary with histogram metrics
    """
    # Basic statistics
    non_zero = histogram[histogram > 0]
    
    metrics = {
        "mean": float(np.mean(histogram)),
        "std_dev": float(np.std(histogram)),
        "max": float(np.max(histogram)),
        "entropy": float(-np.sum(non_zero * np.log2(non_zero + 1e-10))),
        "num_non_zero_bins": int(np.sum(histogram > 0))
    }
    
    # Quantiles
    for q in [25, 50, 75, 90, 95]:
        metrics[f"q{q}"] = float(np.percentile(histogram, q))
    
    return metrics
