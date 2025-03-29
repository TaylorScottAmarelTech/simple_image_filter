"""
Command-line interface for simple_image_filter.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .analyzer import is_image_ok


def main():
    """Run the CLI application."""
    parser = argparse.ArgumentParser(description="Simple Image Filter")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add the 'check' command
    check_parser = subparsers.add_parser("check", help="Check if an image is OK")
    check_parser.add_argument("image", help="Path to image file")
    check_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    check_parser.add_argument("--threshold", type=str, help="Custom thresholds in JSON format")
    
    # Add other subcommands here as needed
    
    args = parser.parse_args()
    
    if args.command == "check":
        try:
            # Open the image
            image_path = Path(args.image)
            if not image_path.exists():
                print(f"Error: Image file not found: {args.image}")
                sys.exit(1)
                
            image = Image.open(args.image)
            
            # Parse custom thresholds if provided
            thresholds = None
            if args.threshold:
                try:
                    thresholds = json.loads(args.threshold)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON format for thresholds: {args.threshold}")
                    sys.exit(1)
            
            # Check image quality
            is_ok, details = is_image_ok(image, thresholds)
            
            if args.json:
                # Add the overall result to the details dictionary
                details["is_ok"] = is_ok
                print(json.dumps(details, indent=2))
            else:
                if is_ok:
                    print("✅ Image is OK")
                else:
                    print("❌ Image is NOT OK")
                
                print("\nDetails:")
                print(f"  Brightness: {details['brightness']:.2f} {'✅' if details['is_brightness_ok'] else '❌'}")
                print(f"  Contrast: {details['contrast']:.2f} {'✅' if details['is_contrast_ok'] else '❌'}")
                print(f"  Saturation: {details['saturation']:.2f} {'✅' if details['is_saturation_ok'] else '❌'}")
                print(f"  Sharpness: {details['sharpness']:.2f} {'✅' if details['is_sharpness_ok'] else '❌'}")
            
            # Return exit code based on result (0 for OK, 1 for not OK)
            sys.exit(0 if is_ok else 1)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()