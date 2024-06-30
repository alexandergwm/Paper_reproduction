import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader

def minmaxscaler(data):
    min = data.min()
    max = data.max()    
    return (data)/(max-min)


def load_weight_for_model(model, pretrained_path, device):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location= device)

    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]

    model.load_state_dict(model_dict)

class OneD_CNN_Pre(nn.Module):
    """
    1D CNN, get the frequency features of the signal
    """
    def __init__(self):
        super().__init__()
        # First layer
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels = 10,
            kernel_size = 3,
            stride = 1
        )
        # Secondary layer
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

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        logit = x.view(x.shape[0], -1)

        return logit

class ONED_CNN_Predictor():
    
    def __init__(self, MODEL_PTH, device):
        self.cnn = OneD_CNN_Pre().to(device)
        load_weight_for_model(self.cnn, MODEL_PTH, device)
        self.cnn.eval()
        self.cos = nn.CosineSimilarity(dim=1).to(device)
        self.device = device

    def cosSimilarity(self, signal_1, signal_2):
        signal_1, signal_2 = signal_1.unsqueeze(0), signal_2.unsqueeze(0)
        similarity = self.cos(self.cnn(signal_1.to(self.device)), self.cnn(signal_2.to(self.device)))
        return similarity.cpu().item()
    
    def cosSimilarity_minmax(self, signal_1, signal_2):
        signal_1, signal_2 = minmaxscaler(signal_1).unsqueeze(0), minmaxscaler(signal_2).unsqueeze(0)
        similarity = self.cos(self.cnn(signal_1.to(self.device)), self.cnn(signal_2.to(self.device)))
        return similarity.cpu().item()


#------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "feedforwardnet.pth"
    cnn = OneD_CNN_Pre().to(device)
    
    summary(cnn, (1,16000))