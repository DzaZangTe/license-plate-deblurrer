import torch
from PIL import Image
import os
from torchvision import transforms
from detector import LicensePlateDetector
import time
import numpy as np
import cv2

def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def detect_and_save(model, image_path, output_folder, transform):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    w, h = image.size

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    outputs = outputs.cpu().numpy()[0]
    outputs[0::2] *= w
    outputs[1::2] *= h

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    parts = file_name.replace('&', '_').split('_')
    annotation = list(map(int, parts))
    l2_distance = sum((annotation[i] - outputs[i])**2 for i in range(len(annotation)))
    print(l2_distance)

    corners = [(outputs[i], outputs[i+1]) for i in range(0, len(outputs), 2)]

    image_np = np.array(image)
    warped_image = four_point_transform(image_np, corners)

    warped_image_pil = Image.fromarray(warped_image)
    output_path = os.path.join(output_folder, f"{file_name}.jpg")
    warped_image_pil.save(output_path)
    return l2_distance

time_stamp0 = time.time()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

model = LicensePlateDetector()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load('saved_models/model_epoch_50.pth'))

test_folder = 'testing_data'
output_folder = 'testing_results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

time_stamp1 = time.time()

cnt = 0
total_sum = 0.0
for img_file in os.listdir(test_folder):
    if img_file.endswith('.jpg'):
        cnt += 1
        total_sum += detect_and_save(model, os.path.join(test_folder, img_file), output_folder, transform)

print(f"Average MSE: {total_sum / cnt / 8:.5f}")

time_stamp2 = time.time()

print(f"Model loading time: {time_stamp1 - time_stamp0:.5f}s")
print(f"Detection time per image: {(time_stamp2 - time_stamp1) / cnt:.5f}s")