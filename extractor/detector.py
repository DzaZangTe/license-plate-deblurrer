import torch.nn as nn
import torchvision.models as models

class LicensePlateDetector(nn.Module):
    def __init__(self):
        super(LicensePlateDetector, self).__init__()
        self.features = models.resnet18(pretrained=False)
        self.features.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        return self.features(x)