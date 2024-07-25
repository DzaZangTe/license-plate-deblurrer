import os
import cv2
import numpy as np
import random
import glob
import sys

def add_motion_blur(image, degree, angle):
    image = np.array(image)
    ksize = degree
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
    kernel = cv2.warpAffine(kernel,
                            cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0),
                            (ksize, ksize))
    kernel = kernel / ksize
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def get_corners_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('-')
    coords = parts[3].split('_')
    corners = []
    for coord in coords:
        x, y = map(int, coord.split('&'))
        corners.append([x, y])
    return np.array(corners, dtype='float32')

def compute_new_corners(corners, degree, angle):
    rad = np.deg2rad(angle)
    dx = degree * np.cos(rad) / 2
    dy = -degree * np.sin(rad) / 2
    
    new_corners = corners.copy()
    new_corners[:, 0] += dx
    new_corners[:, 1] += dy
    return new_corners

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def compute_intersection_points(corners1, corners2):
    lines1 = [(corners1[i], corners1[(i+1) % 4]) for i in range(4)]
    lines2 = [(corners2[i], corners2[(i+1) % 4]) for i in range(4)]

    intersection_points = []
    for i in range(4):
        for j in range(4):
            if i % 2 == j % 2:
                continue
            point = line_intersection(lines1[i], lines2[j])
            if point is not None:
                intersection_points.append(point)

    return np.array(intersection_points, dtype='float32')

def compute_convex_hull(points):
    hull = cv2.convexHull(points)
    hull = hull.reshape(-1, 2)
    hull_list = hull.tolist()
    hull_list.append(hull_list[0])
    hull_list.append(hull_list[1])
    hull = np.array(hull_list, dtype='float32')
    ret = []
    for i in range(len(hull) - 2):
        angle1 = np.arctan2(hull[i+1][1] - hull[i][1], hull[i+1][0] - hull[i][0])
        angle2 = np.arctan2(hull[i+2][1] - hull[i+1][1], hull[i+2][0] - hull[i+1][0])
        diff = np.abs(angle1 - angle2)
        diff = np.minimum(diff, 2 * np.pi - diff)
        if (diff > 0.01):
            ret.append(hull[i+1])
    return np.array(ret, dtype='float32')

def perspective_transform(image, src_points, size=(256, 256)):
    dst_points = np.array([[0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1], [0, size[1]-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, size)
    return warped

if len(sys.argv) >= 6:
    image_folder = sys.argv[1]
    output_folder = sys.argv[2]
    output_folder2 = sys.argv[3]
    output_folder3 = sys.argv[4]
    rate = float(sys.argv[5])
else:
    print("Usage: python extract.py <image_folder> <full_streetscape> <chopped_sharp> <chopped_blur> <rate>")
    sys.exit(1)

image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)
os.makedirs(output_folder3, exist_ok=True)

counter = 1

for selected_image_file in image_files:
    if random.random() > rate:
        continue
    image = cv2.imread(selected_image_file)
    height, width = image.shape[:2]

    degree = random.randint(5, 20)
    angle = random.uniform(-30, 30)
    blurred_image = add_motion_blur(image, degree, angle)

    corners = get_corners_from_filename(selected_image_file)
    new_corners1 = compute_new_corners(corners, degree, angle)
    new_corners2 = compute_new_corners(corners, -degree, angle)
    intersection_points = compute_intersection_points(new_corners1, new_corners2)
    hull_points = compute_convex_hull(np.vstack((new_corners1, new_corners2, intersection_points)))
    idx = np.argmin(hull_points.sum(axis=1))
    hull_points = np.roll(hull_points, -idx, axis=0)

    if (len(hull_points) != 4):
        continue

    new_image_filename = "{}&{}_{}&{}_{}&{}_{}&{}.jpg".format(
        int(hull_points[0][0]), int(hull_points[0][1]),
        int(hull_points[1][0]), int(hull_points[1][1]),
        int(hull_points[2][0]), int(hull_points[2][1]),
        int(hull_points[3][0]), int(hull_points[3][1])
    )
    new_image_filepath = os.path.join(output_folder, new_image_filename)
    cv2.imwrite(new_image_filepath, blurred_image)

    warped_image = perspective_transform(image, hull_points)
    new_image_filepath2 = os.path.join(output_folder2, f'chop{counter}.jpg')
    cv2.imwrite(new_image_filepath2, warped_image)

    warped_image = perspective_transform(blurred_image, hull_points)
    new_image_filepath3 = os.path.join(output_folder3, f'chop{counter}.jpg')
    cv2.imwrite(new_image_filepath3, warped_image)

    counter += 1