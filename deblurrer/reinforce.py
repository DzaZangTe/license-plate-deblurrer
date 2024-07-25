import os
import cv2
import numpy as np
import random

def perspective_transform(image):
    height, width = image.shape[:2]
    src_points = np.float32([
        [random.uniform(0, width * 0.1), random.uniform(0, height * 0.1)],
        [random.uniform(0, width * 0.1), random.uniform(height * 0.9, height)],
        [random.uniform(width * 0.9, width), random.uniform(height * 0.9, height)],
        [random.uniform(width * 0.9, width), random.uniform(0, height * 0.1)]
    ])
    dst_points = np.float32([
        [0, 0],
        [0, height],
        [width, height],
        [width, 0]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))
    return transformed_image

def motion_blur(image):
    size = random.randint(15, 40)
    angle = random.uniform(-30, 30)
    kernel_motion_blur = np.zeros((size, size))
    center = size // 2
    tan_angle = np.tan(np.deg2rad(angle))
    if abs(tan_angle) <= 1:
        for i in range(size):
            offset = int(tan_angle * (i - center))
            weight = (i + 1) if i <= center else (size - i)
            if 0 <= center + offset < size:
                kernel_motion_blur[center + offset, i] = weight
    else:
        cot_angle = 1 / tan_angle
        for i in range(size):
            offset = int(cot_angle * (i - center))
            weight = (i + 1) if i <= center else (size - i)
            if 0 <= center + offset < size:
                kernel_motion_blur[i, center + offset] = weight
    kernel_motion_blur = kernel_motion_blur / kernel_motion_blur.sum()
    blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)
    scale_factor = random.uniform(0.9, 1.1)
    height, width = blurred_image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    resized_image = cv2.resize(blurred_image, (new_width, new_height))
    if scale_factor > 1.0:
        resized_image = resized_image[(new_height - height) // 2: (new_height + height) // 2, 
                                      (new_width - width) // 2: (new_width + width) // 2]
    else:
        padded_image = np.zeros((height, width, 3), dtype=np.uint8)
        y_offset = (height - new_height) // 2
        x_offset = (width - new_width) // 2
        padded_image[y_offset: y_offset + new_height, x_offset: x_offset + new_width] = resized_image
        resized_image = padded_image
    return resized_image

def augment_images(input_blur_dir, input_sharp_dir, output_blur_dir, output_sharp_dir):
    if not os.path.exists(output_blur_dir):
        os.makedirs(output_blur_dir)
    if not os.path.exists(output_sharp_dir):
        os.makedirs(output_sharp_dir)

    for filename in os.listdir(input_blur_dir):
        if filename.startswith('grab'):
            base_name = filename.split('grab')[1]
            blur_image_path = os.path.join(input_blur_dir, filename)
            sharp_image_path = os.path.join(input_sharp_dir, filename)

            blur_image = cv2.imread(blur_image_path)
            sharp_image = cv2.imread(sharp_image_path)

            transformed_blur_image = perspective_transform(blur_image)
            transformed_sharp_image = perspective_transform(sharp_image)

            trans_blur_path = os.path.join(output_blur_dir, f'trans{base_name}')
            trans_sharp_path = os.path.join(output_sharp_dir, f'trans{base_name}')

            cv2.imwrite(trans_blur_path, transformed_blur_image)
            cv2.imwrite(trans_sharp_path, transformed_sharp_image)

            motion_blur_image = motion_blur(transformed_sharp_image)
            motion_blur_path = os.path.join(output_blur_dir, f'motion{base_name}')
            motion_sharp_path = os.path.join(output_sharp_dir, f'motion{base_name}')

            cv2.imwrite(motion_blur_path, motion_blur_image)
            cv2.imwrite(motion_sharp_path, transformed_sharp_image)

input_blur_dir = 'training_blur'
input_sharp_dir = 'training_sharp'
output_blur_dir = 'training_blur'
output_sharp_dir = 'training_sharp'

augment_images(input_blur_dir, input_sharp_dir, output_blur_dir, output_sharp_dir)