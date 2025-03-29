# Simple Image Filter

A Python package for detecting and filtering unrealistic images based on RGB histogram analysis. This package identifies images with the characteristic 3-spike histogram pattern with low standard deviation that's common in problematic AI-generated content.

## Features

- Analyzes RGB histograms to detect unrealistic patterns
- Identifies images with the characteristic 3-spike pattern
- Provides detailed analysis of histogram characteristics
- Includes command-line interface for batch processing
- Generates visualizations of histogram analysis

## Installation

Install from PyPI:

pip install simple-image-filter

## Usage

### Python API

from simple_image_filter import ImageAnalyzer

# Initialize analyzer with custom thresholds (optional)
analyzer = ImageAnalyzer(
    peak_threshold=0.1,        # Minimum height for peaks
    std_dev_threshold=15.0,    # Maximum standard deviation for unrealistic images
    max_peaks=3                # Number of peaks to look for
)

# Analyze a single image
result = analyzer.analyze_image("path/to/image.jpg")

# Print analysis results
print(f"Image is realistic: {result['is_realistic']}")
if not result['is_realistic']:
    print(f"Reason: {result['reason']}")

# Quick check for realistic image
is_realistic = analyzer.is_image_realistic("path/to/image.jpg")
