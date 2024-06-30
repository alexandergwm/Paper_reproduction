import torch.nn as nn
import torchvision.models as models

class Modified_ShufflenetV2(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.bw2col = nn.Sequential(
            nn.Conv2d(1, 10, 1, padding=0),  # Conv2d with 1 input channel
            nn.BatchNorm2d(10),              # BatchNorm2d after Conv2d
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 3, 1, padding=0),  # Conv2d with 10 input channels
            nn.ReLU(inplace=True))

        self.mv2 = models.shufflenet_v2_x0_5(pretrained=True)  # Pre-trained ShuffleNetV2

        # Modify conv5 layer to change output channels from 192 to 512
        self.mv2.conv5 = nn.Sequential(
            nn.Conv2d(192, 512, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        # Modify fully connected layer (fc) to output num_classes
        self.mv2.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.bw2col(x)
        x = self.mv2(x)
        return x