from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path('./data')

IMAGES_ROOT = DATA_ROOT / 'images'
MASKS_DIR = DATA_ROOT / 'car_masks'
SHADOW_DIR = DATA_ROOT / 'shadow_masks'

def load_image_and_masks(image_path):
    """
    Loads an image and its corresponding masks.

    Args:
        image_path (Path): The path to the input image.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The loaded image.
            - numpy.ndarray: The car mask.
            - numpy.ndarray: The shadow mask.
    """
    image_name = image_path.stem
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(MASKS_DIR / f'{image_name}.png'), cv2.IMREAD_GRAYSCALE)
    shadow_mask = cv2.imread(str(SHADOW_DIR / f'{image_name}.png'), cv2.IMREAD_GRAYSCALE)
    
    # Ensure mask has the same dimensions as the image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return image, mask, shadow_mask

def clean_mask(mask, kernel_size=7):
    """
    Cleans noise from binary masks.

    Args:
        mask (numpy.ndarray): Original mask to process.
        kernel_size (int, optional): Size of the kernels for morphological operations. Defaults to 7.

    Returns:
        numpy.ndarray: Processed binary mask with noise and holes removed.
    """
    _, binary_mask = cv2.threshold(mask, 0.498, 1, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel) # morphological opening to remove small noise (isolated pixels)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # morphological closing to remove small holes
    binary_mask = cv2.medianBlur(binary_mask, ksize=5)
    return binary_mask

def detect_background(image, axis=1):
    """
    Detects the starting and ending rows of wall/floor in the image.

    Args:
        image (numpy.ndarray): The input image.
        axis (int, optional): The axis along which to detect the background. Defaults to 1.

    Returns:
        tuple or None: A tuple containing the start and end row indices of the detected background,
                       or None if no background is detected.
    """
    # Check if the image has an alpha channel
    if image.shape[2] == 4:
        # Use alpha channel for detection
        non_transparent = image[:, :, 3] > 0
    else:
        # For RGB images, detect non-white pixels
        non_white = np.any(image != [255, 255, 255], axis=2)
    
    non_empty_rows = np.where(np.any(non_transparent if image.shape[2] == 4 else non_white, axis=axis))[0]
    if len(non_empty_rows) == 0:
        return None  # No ROI detected
    
    start_row = non_empty_rows[0]
    end_row = non_empty_rows[-1]
    
    return start_row, end_row

def resize_keep_aspect_ratio(image, size):
    """
    Resizes an image while keeping the aspect ratio.

    Args:
        image (numpy.ndarray): The input image.
        size (tuple): The target size (width, height).

    Returns:
        numpy.ndarray: The resized image.
    """
    h, w = image.shape[:2]
    ratio = min(size[0] / w, size[1] / h)
    return cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

def overlay_transparent(background, overlay):
    """
    Overlays a transparent overlay on a background image.

    Args:
        background (numpy.ndarray): The background image.
        overlay (numpy.ndarray): The overlay image with an alpha channel.

    Returns:
        numpy.ndarray: The resulting image with the overlay applied.
    """
    mask = overlay[:, :, 3]
    mask = clean_mask(mask)
    background_inv_mask = cv2.bitwise_and(background, background, mask=(1-mask))
    overlay_extracted = cv2.bitwise_and(overlay[:, :, :3], overlay[:, :, :3], mask=mask)
    cv2.imwrite('background_inv_mask.png', background_inv_mask)
    return cv2.add(background_inv_mask[:, :, :3], overlay_extracted)
