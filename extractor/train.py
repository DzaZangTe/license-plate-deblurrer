import os
import json
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from licensePlate import LicensePlateDataset
from detector import LicensePlateDetector

def train(resume_epoch=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = LicensePlateDataset(folder_path='training_data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = LicensePlateDetector().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if resume_epoch > 0:
        model.load_state_dict(torch.load(f'saved_models/model_epoch_{resume_epoch}.pth'))
        print(f"Model loaded from epoch {resume_epoch}")

    num_epochs = 50
    save_interval = 5
    save_path = "saved_models"
    os.makedirs(save_path, exist_ok=True)
    
    loss_dict = {'train_loss': []}

    for epoch in range(resume_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(dataloader)}] [Loss: {loss.item()}]")

        epoch_loss = running_loss / len(dataloader)
        loss_dict['train_loss'].append(epoch_loss)

        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
            print(f"Model saved at epoch {epoch+1}")
    
    with open('training_loss.json', 'w') as f:
        json.dump(loss_dict, f)

if __name__ == '__main__':
    train(resume_epoch=0)