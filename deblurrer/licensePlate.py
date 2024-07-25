import os
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset

class LicensePlateDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, transform=None):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.image_names = os.listdir(blur_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        blur_image_path = os.path.join(self.blur_dir, self.image_names[idx])
        sharp_image_path = os.path.join(self.sharp_dir, self.image_names[idx])

        blur_image = Image.open(blur_image_path).convert("L")
        sharp_image = Image.open(sharp_image_path).convert("L")
        blur_image = ImageOps.equalize(blur_image)
        sharp_image = ImageOps.equalize(sharp_image)

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image, img_name
