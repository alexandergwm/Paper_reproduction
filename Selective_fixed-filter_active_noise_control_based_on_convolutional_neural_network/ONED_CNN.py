import torch
from torch import nn
from torchsummary import summary

from torch.utils.data import DataLoader

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader


class OneD_CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # first layer
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels = 10,
            kernel_size = 3,
            stride = 1
        )
        # secondary layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels = 10,
                out_channels = 20,
                kernel_size = 32,
                stride = 1
            ),
        # Max-pool
            nn.MaxPool1d(
                kernel_size = 512,
                stride = 512
            )
        )
        # Normalized layer
        self.normalized = nn.BatchNorm1d(
            num_features = 620
        )
        # Linear layer 1
        self.linear1 = nn.Linear(
            in_features = 620,
            out_features = 15
        )
        # Active function
        self.active = nn.Tanh()
        # Linear layer 2
        self.linear2 = nn.Linear(
            in_features = 15,
            out_features = 15
        )

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.normalized(x)
        x = self.linear1(x)
        x = self.active(x)
        logits = self.linear2(x)

        return logits

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.relu = nn.ReLU()
        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.res(x)
        out += identity
        return self.relu(out)

class OneD_CNN_with_Res(nn.Module):
    def __init__(self, num_classes=15):
        super(OneD_CNN_with_Res, self).__init__()
        # First layer 
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1)
        # Secondary layer 
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=32, stride=1),
            nn.MaxPool1d(kernel_size=512, stride=512)
        )
        # Adding ResBlock
        self.res_block = ResBlock(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1)

        # Normalized layer 
        self.normalized = nn.BatchNorm1d(num_features=620)
        # Linear layer 1
        self.linear1 = nn.Linear(in_features=620, out_features=15)
        # Active function 
        self.active = nn.Tanh()
        # Linear layer 2
        self.linear2 = nn.Linear(in_features=15, out_features=15)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.res_block(x)
        x = x.view(x.shape[0], -1)
        x = self.normalized(x)
        x = self.linear1(x)
        x = self.active(x)
        logits = self.linear2(x)
        return logits

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    model = OneD_CNN_with_Res().to(device)
    summary(model,(1,16000))
