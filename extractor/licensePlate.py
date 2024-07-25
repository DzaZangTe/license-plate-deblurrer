import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class LicensePlateDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name)
        label = self.parse_label(self.image_files[idx], image.width, image.height)
        if self.transform:
            image = self.transform(image)
        return image, label

    def parse_label(self, filename, width, height):
        filename = os.path.splitext(filename)[0]
        parts = filename.replace('&', '_').split('_')
        coords = list(map(int, parts))
        coords = [coords[i:i+2] for i in range(0, len(coords), 2)]
        coords = [(x / width, y / height) for x, y in coords]
        coords = [coord for xy in coords for coord in xy]
        return torch.tensor(coords, dtype=torch.float)