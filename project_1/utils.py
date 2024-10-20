import cv2
import numpy as np

def clean_mask(mask, kernel_size=7):
    '''
    Cleans noise from binary masks.
    Inputs:
        mask(np.ndarray): Original mask to process
        kernel_size(int): Size of the kernels for morphological operations
    Output:
        mask(np.ndarray): Processed binary mask with noise and holes removal
    '''
    _, mask = cv2.threshold(mask, 0.498, 1, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # morphological opening to remove small noise (isolated pixels)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # morphological closing to remove small holes
    mask = cv2.medianBlur(mask, ksize=5)
    return mask



