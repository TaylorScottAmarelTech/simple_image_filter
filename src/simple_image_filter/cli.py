"""
Command-line interface for the simple-image-filter package.
"""

import os
import sys
import argparse
import json
import logging
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .analyzer import ImageAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze images for unrealistic histogram patterns"
    )
    
    parser.add_argument(
        "input",
        help="Path to image file or directory containing images"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Directory to save analysis results and visualizations",
        default="./image_analysis_results"
    )
    
    parser.add_argument(
        "--visualize",
        help="Generate histogram visualizations",
        action="store_true"
    )
    
    parser.add_argument(
        "--threshold",
        help="Standard deviation threshold (default: 15.0)",
        type=float,
        default=15.0
    )
    
    parser.add_argument(
        "--peak-threshold",
        help="Peak height threshold (default: 0.1)",
        type=float,
        default=0.1
    )
    
    return parser.parse_args()

def visualize_histograms(image_path: str, analysis: dict, output_dir: str) -> None:
    """
    Create and save histogram visualizations.
    
    Args:
        image_path: Path to the original image
        analysis: Analysis results from ImageAnalyzer
        output_dir: Directory to save visualizations
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image for visualization: {image_path}")
        return
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Display original image
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")
    
    # Plot histograms for each channel
    colors = ["blue", "green", "red"]
    for i, color in enumerate(colors):
        channel_stats = analysis["channel_stats"][color]
        histogram = np.array(channel_stats["histogram"])
        peaks = np.array(channel_stats["peaks"])
        
        # Plot histogram
        x = np.arange(len(histogram))
        axs[0, 1].plot(x, histogram, color=color, alpha=0.7, 
                      label=f"{color.capitalize()} (std={channel_stats['std_dev']:.2f})")
        
        # Mark peaks
        if len(peaks) > 0:
            axs[0, 1].plot(peaks, histogram[peaks], "x", color=color, markersize=10)
    
    axs[0, 1].set_title("RGB Histograms")
    axs[0, 1].legend()
    axs[0, 1].set_xlabel("Pixel Value")
    axs[0, 1].set_ylabel("Frequency")
    
    # Display individual channels
    for i, color in enumerate(["blue", "green", "red"]):
        row, col = (1, i % 2) if i < 2 else (1, 1)
        
        # Extract the channel (accounting for BGR order in OpenCV)
        channel_idx = {"blue": 0, "green": 1, "red": 2}[color]
        channel = img[:, :, channel_idx]
        
        # Plot histogram for this channel
        if i < 2:  # Only show first 2 channels in individual plots
            hist = np.array(analysis["channel_stats"][color]["histogram"])
            axs[row, col].plot(np.arange(len(hist)), hist, color=color)
            axs[row, col].set_title(f"{color.capitalize()} Channel Histogram")
            
            # Mark peaks
            peaks = np.array(analysis["channel_stats"][color]["peaks"])
            if len(peaks) > 0:
                axs[row, col].plot(peaks, hist[peaks], "rx")
                
            # Add annotations
            is_realistic = not (analysis["channel_stats"][color]["has_spike_pattern"] and 
                             analysis["channel_stats"][color]["std_dev"] < 15.0)
            status = "Realistic" if is_realistic else "Unrealistic"
            axs[row, col].text(0.05, 0.95, f"Status: {status}\nStd Dev: {analysis['channel_stats'][color]['std_dev']:.2f}",
                             transform=axs[row, col].transAxes, 
                             verticalalignment="top",
                             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # Set overall title
    realistic_status = "REALISTIC" if analysis["is_realistic"] else "UNREALISTIC"
    plt.suptitle(f"Histogram Analysis: {os.path.basename(image_path)} - {realistic_status}", 
                 fontsize=16, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_histogram_analysis.png")
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")

def analyze_images(input_path: str, output_dir: str, visualize: bool, 
                 std_dev_threshold: float, peak_threshold: float) -> None:
    """
    Analyze images and save results.
    
    Args:
        input_path: Path to image or directory
        output_dir: Directory to save results
        visualize: Whether to generate visualizations
        std_dev_threshold: Threshold for standard deviation
        peak_threshold: Threshold for peak detection
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ImageAnalyzer(
        peak_threshold=peak_threshold,
        std_dev_threshold=std_dev_threshold
    )
    
    # Process single image or directory
    if os.path.isfile(input_path):
        image_paths = [input_path]
    elif os.path.isdir(input_path):
        # Get all image files in directory
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        image_paths = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f)) and 
            os.path.splitext(f)[1].lower() in extensions
        ]
    else:
        logger.error(f"Input path not found: {input_path}")
        return
    
    # Process each image
    results = []
    for img_path in image_paths:
        try:
            logger.info(f"Analyzing {img_path}")
            analysis = analyzer.analyze_image(img_path)
            
            # Add image path to results
            result = {
                "image_path": img_path,
                "filename": os.path.basename(img_path),
                "is_realistic": analysis["is_realistic"],
                "reason": analysis["reason"]
            }
            results.append(result)
            
            # Save full analysis as JSON
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
            with open(json_path, "w") as f:
                json.dump(analysis, f, indent=2)
            
            # Generate visualization if requested
            if visualize:
                visualize_histograms(img_path, analysis, output_dir)
                
        except Exception as e:
            logger.error(f"Error analyzing {img_path}: {str(e)}")
    
    # Save summary results
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    realistic_count = sum(1 for r in results if r["is_realistic"])
    unrealistic_count = len(results) - realistic_count
    
    print("\n====== Analysis Complete ======")
    print(f"Total images analyzed: {len(results)}")
    print(f"Realistic images: {realistic_count}")
    print(f"Unrealistic images: {unrealistic_count}")
    print(f"Results saved to: {output_dir}")
    
    # List unrealistic images
    if unrealistic_count > 0:
        print("\nUnrealistic images:")
        for r in results:
            if not r["is_realistic"]:
                print(f"- {r['filename']}: {r['reason']}")

def main() -> None:
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    try:
        analyze_images(
            args.input,
            args.output,
            args.visualize,
            args.threshold,
            args.peak_threshold
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
