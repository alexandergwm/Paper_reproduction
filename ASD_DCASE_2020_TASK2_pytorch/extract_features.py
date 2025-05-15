from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import os
from parameters import params
import torch
import logging
from utils import *
from tqdm import tqdm
import librosa
import sys
from sklearn.model_selection import train_test_split
import h5py

warnings.filterwarnings("ignore")

class FeatureExtractor:
    def __init__(self, params):
        """
        Initialize feature extraction class
        
        Args:
            params (dict): Dictionary containing all configuration parameters
        """
        self.params = params
        self.data_root = Path(params['dev_dir'])
        self.save_to = Path(params['feat_dir'])
        self.types = [t.name for t in sorted(self.data_root.glob('*')) if t.is_dir()]
        self.df = None
        
        # Setup logging
        self.setup_logging()
        
        # Audio processing parameters
        self.sample_rate = params['sample_rate']
        self.n_fft = params['n_fft']
        self.hop_length = params['hop_length']
        self.n_mels = params['n_mels']
        self.power = params['power']
        self.n_frames = params['n_frames']
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.params['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'feature_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def extract_multiframe_features(self, file_name):
        """
        Extract multiframe log-mel spectrogram features from audio file.
        
        Args:
            file_name (str): Path to the audio file
            
        Returns:
            np.array: Vector array of shape (n_frames, n_mels * frames)
                     Each row is a concatenated feature vector of multiple frames
        """
        dims = self.n_mels * self.n_frames
        y, sr = librosa.load(file_name, sr=None)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=self.power)
        log_mel = 20.0 / self.power * np.log10(mel + sys.float_info.epsilon)
        vector_array_size = log_mel.shape[1] - self.n_frames + 1
        if vector_array_size < 1:
            return np.empty((0, dims))
        vector_array = np.zeros((vector_array_size, dims))
        for t in range(self.n_frames):
            vector_array[:, self.n_mels * t: self.n_mels * (t + 1)] = log_mel[:, t: t + vector_array_size].T
        return vector_array
        
    def extract_and_save_all_h5(self, val_ratio=0.1, random_seed=42):
        """
        用 h5py 存储大特征矩阵，避免内存溢出。train/val/test 分别存为 train.h5, val.h5, test.h5
        """
        save_dir = self.save_to
        save_dir.mkdir(parents=True, exist_ok=True)
        train_h5 = save_dir / 'train.h5'
        val_h5 = save_dir / 'val.h5'
        test_h5 = save_dir / 'test.h5'

        # 判断文件是否都已存在
        if train_h5.exists() and val_h5.exists() and test_h5.exists():
            self.logger.info('train.h5, val.h5, test.h5 已存在，跳过特征提取。')
            return

        self.df = pd.DataFrame()
        self.df['file'] = sorted(self.data_root.glob('*/*/*.wav'))
        self.df['type'] = self.df.file.map(lambda f: f.parent.parent.name)
        self.df['split'] = self.df.file.map(lambda f: f.parent.name)
        self.logger.info(f'Data preparation completed. Machine types: {self.types}')

        # 1. 收集文件和标签
        train_files, train_labels = [], []
        test_files, test_labels = [], []
        for machine_type in self.types:
            for split in ['train', 'test']:
                type_df = self.df[
                    (self.df['type'] == machine_type) & 
                    (self.df['split'] == split)
                ].reset_index()
                if len(type_df) == 0:
                    continue
                self.logger.info(f'收集 {machine_type} - {split} ({len(type_df)} files)')
                for audio_file in tqdm(type_df['file'], desc=f'{machine_type}-{split}'):
                    if 'normal' in audio_file.name:
                        label = 0
                    elif 'anomaly' in audio_file.name:
                        label = 1
                    else:
                        label = -1
                    if split == 'train':
                        train_files.append(audio_file)
                        train_labels.append(label)
                    else:
                        test_files.append(audio_file)
                        test_labels.append(label)

        # 2. 划分 train/val
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files, train_labels, test_size=val_ratio, random_state=random_seed, stratify=train_labels
        )

        # 3. 统计帧数
        def count_total_frames(file_list):
            total = 0
            for f in tqdm(file_list, desc='统计帧数'):
                features = self.extract_multiframe_features(str(f))
                total += features.shape[0]
            return total

        feature_dim = self.n_mels * self.n_frames
        self.logger.info('统计 train 总帧数...')
        train_total = count_total_frames(train_files)
        self.logger.info('统计 val 总帧数...')
        val_total = count_total_frames(val_files)
        self.logger.info('统计 test 总帧数...')
        test_total = count_total_frames(test_files)

        # 1. 统计均值
        sum_ = 0
        count = 0
        for f in tqdm(train_files, desc='统计train特征均值'):
            features = self.extract_multiframe_features(str(f))
            if features.shape[0] > 0:
                sum_ += features.sum(axis=0)
                count += features.shape[0]
        mean = sum_ / count

        # 2. 统计方差
        sq_sum = 0
        for f in tqdm(train_files, desc='统计train特征方差'):
            features = self.extract_multiframe_features(str(f))
            if features.shape[0] > 0:
                sq_sum += ((features - mean) ** 2).sum(axis=0)
        std = np.sqrt(sq_sum / count) + 1e-8

        # 保存mean和std，方便后续推理用
        np.save(self.save_to / 'feature_mean.npy', mean)
        np.save(self.save_to / 'feature_std.npy', std)

        # 写入h5时归一化
        def save_to_h5py(file_list, label_list, h5_path, total_frames):
            with h5py.File(h5_path, 'w') as h5f:
                features_ds = h5f.create_dataset('features', shape=(total_frames, feature_dim), dtype='float32')
                labels_ds = h5f.create_dataset('labels', shape=(total_frames,), dtype='int64')
                idx = 0
                for f, label in tqdm(zip(file_list, label_list), total=len(file_list), desc=f'写入{h5_path.name}'):
                    features = self.extract_multiframe_features(str(f))
                    if features.shape[0] == 0:
                        continue
                    # 归一化
                    features = (features - mean) / std
                    n = features.shape[0]
                    features_ds[idx:idx+n] = features
                    labels_ds[idx:idx+n] = label
                    idx += n
                if idx < total_frames:
                    features_ds.resize((idx, feature_dim))
                    labels_ds.resize((idx,))
            self.logger.info(f'{h5_path} 写入完成，总帧数: {idx}')

        self.logger.info('写入 train.h5...')
        save_to_h5py(train_files, train_labels, train_h5, train_total)
        self.logger.info('写入 val.h5...')
        save_to_h5py(val_files, val_labels, val_h5, val_total)
        self.logger.info('写入 test.h5...')
        save_to_h5py(test_files, test_labels, test_h5, test_total)
        self.logger.info('全部特征已分批保存为 train/val/test.h5')
    
def main():
    # Create feature extractor instance
    extractor = FeatureExtractor(params)
    extractor.extract_and_save_all_h5()

if __name__ == "__main__":
    main()
