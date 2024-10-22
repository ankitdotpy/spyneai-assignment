import os
from pathlib import Path

import cv2
import numpy as np
from utils import detect_background

DATA_DIR = Path("./data")
IMAGES_DIR = DATA_DIR / "images"
CAR_MASKS_DIR = DATA_DIR / "car_masks"
SHADOW_MASKS_DIR = DATA_DIR / "shadow_masks"

def remove_background(car_image, car_mask):
    """
    Remove the background from the car image using the car mask.

    :param car_image: Car image
    :param car_mask: Car mask
    :return: Car image without background
    """
    _, binary_mask = cv2.threshold(car_mask, 0.498, 1, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8) # 7x7 seems to work well
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.medianBlur(binary_mask, ksize=5)
    return cv2.bitwise_and(car_image, car_image, mask=binary_mask)

def align_shadow(car_mask, shadow_mask):
    """
    Align the shadow mask to the car mask.

    :param car_mask: Car mask
    :param shadow_mask: Shadow mask
    :return: Aligned shadow mask
    """
    car_contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(car_contours, key=cv2.contourArea) # assuming largest contour is the car
    x, y, w, h = cv2.boundingRect(largest_contour)

    # resize shadow mask to match the width of the car
    shadow_mask = cv2.resize(shadow_mask, (w, shadow_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # create a blank mask the same size as the car mask
    aligned_shadow = np.zeros_like(car_mask)

    # calculate the vertical position to place the shadow (at the bottom of the car)
    shadow_y = y + h - shadow_mask.shape[0]

    # place the resized shadow in the correct position
    aligned_shadow[shadow_y:shadow_y+shadow_mask.shape[0], x:x+w] = shadow_mask
    return aligned_shadow

def center_car(car_image, car_mask, background):
    # Find the contour of the car
    car_contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(car_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the center position in the background
    bg_h, bg_w = background.shape[:2]
    center_x = (bg_w - w) // 2
    center_y = bg_h - h - 20

    # Create a translation matrix
    M = np.float32([[1, 0, center_x - x], [0, 1, center_y - y]])

    # Apply the translation to the car image and mask
    centered_car = cv2.warpAffine(car_image, M, (bg_w, bg_h))
    centered_mask = cv2.warpAffine(car_mask, M, (bg_w, bg_h))

    return centered_car, centered_mask

def create_background(floor, wall):
    """
    Creates a background by combining floor and wall images.

    Args:
        floor (numpy.ndarray): The floor image.
        wall (numpy.ndarray): The wall image.

    Returns:
        numpy.ndarray: The combined background image.
    """
    start_row_floor, end_row_floor = detect_background(floor)
    start_row_wall, end_row_wall = detect_background(wall)
    background = np.vstack((wall[int(end_row_wall-0.7*end_row_wall):end_row_wall, :],
                            floor[start_row_floor:end_row_floor, :]))
    return background

def compose_image(car_image, car_mask, shadow_mask, background, shadow_strength=0.9):
    # Remove background from car image
    car_image_without_bg = remove_background(car_image, car_mask)

    # Center the car in the background
    centered_car, centered_mask = align_car_with_floor(car_image_without_bg, car_mask, background)

    # Align and center the shadow
    aligned_shadow = align_shadow(centered_mask, shadow_mask)
    centered_shadow, _ = align_car_with_floor(aligned_shadow, centered_mask, background)

    # Combine car with background
    result = background.copy()
    car_area = (centered_car.sum(axis=2) != 0)
    result[car_area] = centered_car[car_area]

    # Apply shadow
    shadow_area = (centered_shadow > 0)
    result[shadow_area] = result[shadow_area] * (1 - shadow_strength * centered_shadow[shadow_area, np.newaxis] / 255)

    return result

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

def align_car_with_floor(car_image, car_mask, background):
    car_contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(car_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    bg_h, bg_w = background.shape[:2]
    
    # Assuming the floor starts at 2/5 of the background height
    floor_start = int(bg_h * 0.95)
    
    # Place the car at the bottom of the background
    target_x = (bg_w - w) // 2
    target_y = (floor_start - h) # Place the bottom of the car at the floor start
    M = np.float32([[1, 0, target_x - x], [0, 1, target_y - y]])
    aligned_car = cv2.warpAffine(car_image, M, (bg_w, bg_h))
    aligned_mask = cv2.warpAffine(car_mask, M, (bg_w, bg_h))
    return aligned_car, aligned_mask

def main(image_id):
    car_image = cv2.imread(str(IMAGES_DIR / f"{image_id}.jpeg"))
    car_mask = cv2.imread(str(CAR_MASKS_DIR / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    shadow_mask = cv2.imread(str(SHADOW_MASKS_DIR / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    floor = cv2.imread(str(DATA_DIR / "floor.png"))
    wall = cv2.imread(str(DATA_DIR / "wall.png"))

    background = create_background(floor, wall)
    background = resize_keep_aspect_ratio(background, (1920, 1080))

    result = compose_image(car_image, car_mask, shadow_mask, background)
    cv2.imwrite(f"data/output/{image_id}.png", result)

if __name__ == "__main__":
    for image_id in range(1,7):
        main(image_id)
