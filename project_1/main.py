import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_image_and_masks,
    detect_background,
    clean_mask
)

DATA_ROOT = Path('./data')
IMAGES_ROOT = DATA_ROOT / 'images'
MASKS_DIR = DATA_ROOT / 'car_masks'
SHADOW_DIR = DATA_ROOT / 'shadow_masks'
OUTPUT_DIR = DATA_ROOT / 'output'

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

def extract_car_image(image, mask):
    mask = clean_mask(mask)
    extracted_car = cv2.bitwise_and(image, image, mask=mask)
    car_mask_reshaped = np.expand_dims(mask, axis=-1)
    extracted_car = np.concatenate((image, car_mask_reshaped), axis=-1)
    return extracted_car

def create_background(floor, wall):
    start_row_floor, end_row_floor = detect_background(floor)
    start_row_wall, end_row_wall = detect_background(wall)
    background = np.vstack((wall[int(end_row_wall-0.7*end_row_wall):end_row_wall, :],
                            floor[start_row_floor:end_row_floor, :]))
    return background

def extract_car_images(car_image, car_mask):
    car_mask = clean_mask(car_mask)
    extracted_car = extract_car_image(car_image, car_mask)
    return extracted_car

def resize_keep_aspect_ratio(image, size):
    h, w = image.shape[:2]
    ratio = min(size[0] / w, size[1] / h)
    return cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

def overlay_transparent(background, overlay):
    mask = overlay[:, :, 3]
    mask = clean_mask(mask)
    background_inv_mask = cv2.bitwise_and(background, background, mask=(1-mask))
    overlay_extracted = cv2.bitwise_and(overlay[:, :, :3], overlay[:, :, :3], mask=mask)
    cv2.imwrite('background_inv_mask.png', background_inv_mask)
    return cv2.add(background_inv_mask[:, :, :3], overlay_extracted)


def compose_image(image_path):
    car_image, car_mask, shadow_mask = load_image_and_masks(image_path)
    floor = cv2.imread(DATA_ROOT / 'floor.png', cv2.IMREAD_UNCHANGED)
    wall = cv2.imread(DATA_ROOT / 'wall.png', cv2.IMREAD_UNCHANGED)

    # combine floor and wall to create background
    background = create_background(floor, wall)
    background = resize_keep_aspect_ratio(background, (1920, 1080))

    # extract the car from the image
    extracted_car = extract_car_images(car_image, car_mask)

    st_row, end_row = detect_background(extracted_car, axis=1)
    st_col, end_col = detect_background(extracted_car, axis=0)
    extracted_car = extracted_car[st_row:end_row+10, st_col:end_col, :]
    if extracted_car.shape[0] < 700 and extracted_car.shape[1] < 1200:
        print(image_path, extracted_car.shape)
        extracted_car = resize_keep_aspect_ratio(extracted_car, (1200, 700))
    
    y_offset = int(background.shape[0] * 0.95)
    x_offset = (background.shape[1] - extracted_car.shape[1]) // 2

    # overlay the extracted car on the background
    result = background.copy()
    roi = result[y_offset-extracted_car.shape[0]:y_offset, x_offset:x_offset+extracted_car.shape[1], :]
    roi = overlay_transparent(roi, extracted_car)
    result[y_offset-extracted_car.shape[0]:y_offset, x_offset:x_offset+extracted_car.shape[1], :3] = roi

    # add shadow to the bottom of the car

    # Save the result
    cv2.imwrite(OUTPUT_DIR / f'{image_path.stem}.png', result)

if __name__ == '__main__':
    for image_path in IMAGES_ROOT.glob('*.jpeg'):
        compose_image(image_path)
