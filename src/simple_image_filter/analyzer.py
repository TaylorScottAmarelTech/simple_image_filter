import numpy as np
from PIL import Image, ImageStat

def calculate_brightness(image_array):
    """
    Calculate the perceived brightness of an image.
    Returns a value between 0 (completely black) and 255 (completely white).
    """
    # Convert to grayscale if it's a color image
    if len(image_array.shape) == 3:
        # Use standard luminance formula
        r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = image_array
    
    return np.mean(gray)

def calculate_contrast(image_array):
    """
    Calculate the contrast of an image.
    Returns the standard deviation of pixel values.
    """
    # Convert to grayscale if it's a color image
    if len(image_array.shape) == 3:
        r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = image_array
    
    return np.std(gray)

def calculate_saturation(image_array):
    """
    Calculate the average saturation of an image.
    Returns a value between 0 (grayscale) and 1 (fully saturated).
    """
    # Only applicable to color images
    if len(image_array.shape) < 3 or image_array.shape[2] < 3:
        return 0
    
    # Convert RGB to HSV and extract saturation channel
    from skimage.color import rgb2hsv
    hsv = rgb2hsv(image_array)
    saturation = hsv[:, :, 1]
    
    return np.mean(saturation)

def calculate_sharpness(image_array):
    """
    Estimate image sharpness using Laplacian variance.
    Higher values indicate sharper images.
    """
    from scipy import ndimage
    
    # Convert to grayscale if it's a color image
    if len(image_array.shape) == 3:
        r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = image_array
    
    # Apply Laplacian filter
    laplacian = np.abs(ndimage.laplace(gray))
    
    return np.var(laplacian)

def is_image_ok(image, thresholds=None):
    """
    Primary function to determine if an image is of acceptable quality.
    
    Parameters:
    -----------
    image : PIL.Image.Image or numpy.ndarray
        The image to analyze
    thresholds : dict, optional
        Dictionary of threshold values for various metrics.
        If None, default thresholds will be used.
        
    Returns:
    --------
    bool
        True if the image passes quality checks, False otherwise
    dict
        Detailed analysis results with individual metric scores
    """
    # Default thresholds
    default_thresholds = {
        'brightness_min': 40,     # Minimum acceptable brightness
        'brightness_max': 220,    # Maximum acceptable brightness
        'contrast_min': 15,       # Minimum acceptable contrast
        'saturation_min': 0.1,    # Minimum acceptable saturation
        'saturation_max': 0.9,    # Maximum acceptable saturation
        'sharpness_min': 100      # Minimum acceptable sharpness
    }
    
    # Use provided thresholds or defaults
    if thresholds is None:
        thresholds = default_thresholds
    else:
        # Update default thresholds with any provided values
        for key, value in thresholds.items():
            default_thresholds[key] = value
        thresholds = default_thresholds
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
        
    # Calculate metrics
    brightness = calculate_brightness(image_array)
    contrast = calculate_contrast(image_array)
    saturation = calculate_saturation(image_array)
    sharpness = calculate_sharpness(image_array)
    
    # Evaluate against thresholds
    is_brightness_ok = thresholds['brightness_min'] <= brightness <= thresholds['brightness_max']
    is_contrast_ok = contrast >= thresholds['contrast_min']
    is_saturation_ok = thresholds['saturation_min'] <= saturation <= thresholds['saturation_max']
    is_sharpness_ok = sharpness >= thresholds['sharpness_min']
    
    # Overall result - image is OK if all checks pass
    overall_result = all([
        is_brightness_ok,
        is_contrast_ok, 
        is_saturation_ok,
        is_sharpness_ok
    ])
    
    # Detailed analysis results
    details = {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'sharpness': sharpness,
        'is_brightness_ok': is_brightness_ok,
        'is_contrast_ok': is_contrast_ok,
        'is_saturation_ok': is_saturation_ok,
        'is_sharpness_ok': is_sharpness_ok
    }
    
    return overall_result, details