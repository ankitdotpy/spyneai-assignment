from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path('./data')

IMAGES_ROOT = DATA_ROOT / 'images'
MASKS_DIR = DATA_ROOT / 'car_masks'
SHADOW_DIR = DATA_ROOT / 'shadow_masks'

def load_image_and_masks(image_path):
    image_name = image_path.stem
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(MASKS_DIR / f'{image_name}.png'), cv2.IMREAD_GRAYSCALE)
    shadow_mask = cv2.imread(str(SHADOW_DIR / f'{image_name}.png'), cv2.IMREAD_GRAYSCALE)
    
    # Ensure mask has the same dimensions as the image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return image, mask, shadow_mask

def clean_mask(mask, kernel_size=7):
    '''
    Cleans noise from binary masks.
    Inputs:
        mask(np.ndarray): Original mask to process
        kernel_size(int): Size of the kernels for morphological operations
    Output:
        mask(np.ndarray): Processed binary mask with noise and holes removal
    '''
    _, binary_mask = cv2.threshold(mask, 0.498, 1, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel) # morphological opening to remove small noise (isolated pixels)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # morphological closing to remove small holes
    binary_mask = cv2.medianBlur(binary_mask, ksize=5)
    return binary_mask

def detect_background(image, axis=1):
    '''
    Detects the starting and ending rows of wall/floor in the image.
    '''
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
