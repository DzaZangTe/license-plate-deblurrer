import os
import re
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
from UNet import UNetGenerator
from licensePlate import LicensePlateDataset
import time
import sys

def find_latest_model(directory):
    max_epoch = -1
    latest_model = None

    for filename in os.listdir(directory):
        match = re.search(r'generator_epoch_(\d+)\.pth', filename)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_model = filename

    return latest_model

def test(model=-1):
    time_stamp0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    test_dataset = LicensePlateDataset(blur_dir='testing_blur', sharp_dir='testing_blur', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if model == -1:
        model = find_latest_model('saved_models')
    else:
        model = f'generator_epoch_{model}.pth'

    print(f'Using model: {model}')

    generator = UNetGenerator(in_channels=1, out_channels=1).to(device)
    generator.load_state_dict(torch.load('saved_models/' + model))
    generator.eval()

    os.makedirs('testing_results', exist_ok=True)

    time_stamp1 = time.time()
    cnt = 0
    
    with torch.no_grad():
        for i, (blur_image, _, img_name) in enumerate(test_dataloader):
            cnt += 1
            blur_image = blur_image.to(device)
            gen_sharp = generator(blur_image)
            resize_transform = transforms.Resize((150, 350))
            gen_sharp_resized = resize_transform(gen_sharp)
            
            save_image(gen_sharp_resized, f'testing_results/{img_name[0]}')
            print(f'Saved: testing_results/{img_name[0]}')
    
    time_stamp2 = time.time()
    print(f"Time for loading model: {time_stamp1 - time_stamp0}")
    print(f"Time for generating each image: {(time_stamp2 - time_stamp1) / cnt}")

if __name__ == '__main__':
    if len(sys.argv) == 2:
        test(int(sys.argv[1]))
    else:
        test()