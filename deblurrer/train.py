import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from UNet import UNetGenerator
from patchGAN import PatchGANDiscriminator
from licensePlate import LicensePlateDataset
import json

def save_losses(G_losses, D_losses, filename='train_losses.json'):
    with open(filename, 'w') as f:
        json.dump({'G_losses': G_losses, 'D_losses': D_losses}, f)

def load_losses(filename='train_losses.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            losses = json.load(f)
            return losses['G_losses'], losses['D_losses']
    else:
        return [], []

def train(resume_epoch=0):
    G_losses, D_losses = load_losses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = LicensePlateDataset(blur_dir='training_blur', sharp_dir='training_sharp', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()

    generator = UNetGenerator(in_channels=1, out_channels=1).to(device)
    discriminator = PatchGANDiscriminator(in_channels=1).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    if resume_epoch > 0:
        generator.load_state_dict(torch.load(f'saved_models/generator_epoch_{resume_epoch}.pth'))
        discriminator.load_state_dict(torch.load(f'saved_models/discriminator_epoch_{resume_epoch}.pth'))
        print(f"Model loaded from epoch {resume_epoch}")

    num_epochs = 100
    for epoch in range(resume_epoch, num_epochs):
        for i, (blur_image, sharp_image, _) in enumerate(dataloader):
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)

            valid = torch.ones((blur_image.size(0), 1, 16, 16)).to(device)
            fake = torch.zeros((blur_image.size(0), 1, 16, 16)).to(device)

            optimizer_G.zero_grad()
            gen_sharp = generator(blur_image)
            loss_GAN = criterion_GAN(discriminator(gen_sharp), valid)
            loss_pixel = criterion_pixel(gen_sharp, sharp_image)
            loss_G = loss_GAN + 100 * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_real = criterion_GAN(discriminator(sharp_image), valid)
            loss_fake = criterion_GAN(discriminator(gen_sharp.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

        os.makedirs('saved_models', exist_ok=True)
        torch.save(generator.state_dict(), f'saved_models/generator_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'saved_models/discriminator_epoch_{epoch+1}.pth')
        save_losses(G_losses, D_losses)
        print(f"Model saved at epoch {epoch+1}")

if __name__ == '__main__':
    train(84)