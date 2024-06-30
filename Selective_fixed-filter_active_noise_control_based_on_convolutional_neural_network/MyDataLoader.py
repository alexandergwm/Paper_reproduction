import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import os

def minmaxscaler(data):
    return (data - data.min()) / (data.max() - data.min())

class MyNoiseDataset(Dataset):
    def __init__(self, folder, annotations_file, use_stft=False, n_fft=256, hop_length=128, db_transform=True):
        self.folder = folder
        self.annotations_file = pd.read_csv(os.path.join(folder, annotations_file))
        self.use_stft = use_stft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.db_transform = db_transform  # 是否转换为分贝值
        self.target_length = hop_length * (124 - 1) + n_fft

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path))
        signal = minmaxscaler(signal)

        if self.use_stft:
            hann_window = torch.hann_window(window_length=self.n_fft, periodic=True, device=signal.device)
            if signal.shape[1] > self.target_length:
                signal = signal[:, :self.target_length]
            elif signal.shape[1] < self.target_length:
                pad_size = self.target_length - signal.shape[1]
                signal = torch.nn.functional.pad(signal, (0, pad_size))
            spectorgram = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=2, center=False, onesided=True)(signal)
            if self.db_transform:
                spectorgram = torchaudio.transforms.AmplitudeToDB()(spectorgram)
            return spectorgram, label
        else:
            return signal, label

    def _get_audio_sample_path(self, index):
        path = self.annotations_file.iloc[index, 1]
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations_file.iloc[index, 2]
        return label
