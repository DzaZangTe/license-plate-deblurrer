import os
import re
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from UNet import UNetGenerator
from patchGAN import PatchGANDiscriminator
from licensePlate import LicensePlateDataset
import json
from torchvision import transforms
import torch.nn as nn

def save_losses(G_losses, D_losses, filename='test_losses.json'):
    with open(filename, 'w') as f:
        json.dump({'G_losses': G_losses, 'D_losses': D_losses}, f)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    test_dataset = LicensePlateDataset(blur_dir='testing_blur', sharp_dir='testing_blur', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()

    G_losses = []
    D_losses = []

    for epoch in range(1, 101):
        gen_model_path = f'saved_models/generator_epoch_{epoch}.pth'
        disc_model_path = f'saved_models/discriminator_epoch_{epoch}.pth'

        if not os.path.exists(gen_model_path) or not os.path.exists(disc_model_path):
            print(f'Model for epoch {epoch} not found, skipping...')
            continue

        generator = UNetGenerator(in_channels=1, out_channels=1).to(device)
        discriminator = PatchGANDiscriminator(in_channels=1).to(device)
        generator.load_state_dict(torch.load(gen_model_path))
        discriminator.load_state_dict(torch.load(disc_model_path))
        generator.eval()
        discriminator.eval()

        total_loss_G = 0.0
        total_loss_D = 0.0
        valid = torch.ones((1, 1, 16, 16)).to(device)
        fake = torch.zeros((1, 1, 16, 16)).to(device)

        with torch.no_grad():
            for i, (blur_image, sharp_image, _) in enumerate(test_dataloader):
                blur_image = blur_image.to(device)
                sharp_image = sharp_image.to(device)

                gen_sharp = generator(blur_image)
                loss_GAN = criterion_GAN(discriminator(gen_sharp), valid)
                loss_pixel = criterion_pixel(gen_sharp, sharp_image)
                loss_G = loss_GAN + 100 * loss_pixel
                total_loss_G += loss_G.item()

                loss_real = criterion_GAN(discriminator(sharp_image), valid)
                loss_fake = criterion_GAN(discriminator(gen_sharp), fake)
                loss_D = (loss_real + loss_fake) / 2
                total_loss_D += loss_D.item()

        avg_loss_G = total_loss_G / len(test_dataloader)
        avg_loss_D = total_loss_D / len(test_dataloader)
        G_losses.append(avg_loss_G)
        D_losses.append(avg_loss_D)

        print(f"[Epoch {epoch}] [G loss: {avg_loss_G}] [D loss: {avg_loss_D}]")

    save_losses(G_losses, D_losses)
    print("Test losses saved to test_losses.json")

if __name__ == '__main__':
    test()
