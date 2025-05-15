import numpy as np
import pandas as pd
from pathlib import Path
import logging
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from typing import Tuple, List, Dict, Optional
from parameters import params
import h5py

class H5FeatureDataset(Dataset):
    """用于加载 h5py 特征文件的数据集"""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.features = self.h5_file['features']
        self.labels = self.h5_file['labels']
        self.length = self.features.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        # 转为 torch tensor
        feature = torch.from_numpy(feature).float()
        label = torch.tensor(label, dtype=torch.long)
        return feature, label

    def __del__(self):
        # 关闭 h5 文件
        try:
            self.h5_file.close()
        except:
            pass

class FeatureDataset(Dataset):
    """特征数据集类，用于加载和处理.npz特征文件"""
    
    def __init__(self, 
                 params: dict,
                 split: str = 'train',
                 transform: Optional[callable] = None):
        """
        初始化数据集
        
        Args:
            params (dict): 参数字典
            split (str): 数据集划分 (train/val/test)
            transform (callable, optional): 数据增强函数
        """
        self.params = params
        self.data_dir = Path(params['feat_dir'])
        self.machine_type = params['target']
        self.split = split
        self.transform = transform
        
        # 设置日志
        self.setup_logging()
        
        # 获取所有特征文件路径
        if split in ['train', 'val']:
            # 训练集和验证集都从train目录加载
            self.feature_files = sorted(self.data_dir.glob(f'{self.machine_type}/train/*.npz'))
        else:
            # 测试集从test目录加载
            self.feature_files = sorted(self.data_dir.glob(f'{self.machine_type}/test/*.npz'))
            
        if len(self.feature_files) == 0:
            raise ValueError(f"在 {self.data_dir}/{self.machine_type}/{split} 中没有找到特征文件")
            
        self.logger.info(f"加载 {self.machine_type} {split} 数据集: {len(self.feature_files)} 个文件")
        
        # 统计正常和异常样本数量
        normal_count = sum(1 for f in self.feature_files if 'normal' in f.name)
        anomaly_count = len(self.feature_files) - normal_count
        self.logger.info(f"正常样本: {normal_count}, 异常样本: {anomaly_count}")
        
        # 计算每个文件中的帧数并存储文件信息
        self.file_info = []
        total_frames = 0
        for file_path in self.feature_files:
            data = np.load(file_path)
            n_frames = len(data['features'])
            self.file_info.append({
                'path': file_path,
                'n_frames': n_frames,
                'label': data['label'],
                'start_idx': total_frames
            })
            total_frames += n_frames
            
        self.total_frames = total_frames
        self.logger.info(f"总帧数: {self.total_frames}")
        
    def setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def __len__(self) -> int:
        """返回数据集大小（总帧数）"""
        return self.total_frames
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (特征, 标签)
        """
        # 找到对应的文件
        file_idx = 0
        while file_idx < len(self.file_info) - 1 and idx >= self.file_info[file_idx + 1]['start_idx']:
            file_idx += 1
            
        file_info = self.file_info[file_idx]
        frame_idx = idx - file_info['start_idx']
        
        # 加载特征文件
        data = np.load(file_info['path'])
        features = data['features'][frame_idx]
        label = file_info['label']
        
        # 转换为torch张量
        features = torch.from_numpy(features).float()
        label = torch.tensor(label, dtype=torch.long)
        
        # 应用数据增强（如果有）
        if self.transform is not None:
            features = self.transform(features)
            
        return features, label

def get_h5_dataloaders(params):
    data_dir = Path(params['feat_dir'])
    batch_size = params.get('batch_size', 512)
    num_workers = params.get('num_workers', 4)
    pin_memory = params.get('pin_memory', True)
    shuffle = params.get('shuffle', True)

    train_set = H5FeatureDataset(data_dir / 'train.h5')
    val_set = H5FeatureDataset(data_dir / 'val.h5')
    test_set = H5FeatureDataset(data_dir / 'test.h5')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

# 使用示例
if __name__ == "__main__":
    # 确保params中包含必要的参数
    required_params = {
        'feat_dir': params.get('feat_dir', 'features'),
        'batch_size': params.get('batch_size', 512),
        'shuffle': params.get('shuffle', True),
        'num_workers': params.get('num_workers', 4),
        'pin_memory': params.get('pin_memory', True),
        'target': params.get('target', 'fan'),
        'validation_split': params.get('validation_split', 0.1),
        'seed': params.get('seed', 42)
    }
    
    # 获取所有数据加载器
    loaders = get_h5_dataloaders(params)
    
    # 测试数据加载
    for split, loader in loaders.items():
        print(f"\n{split} 数据集:")
        for features, labels in loader:
            print(f"特征形状: {features.shape}")
            print(f"标签形状: {labels.shape}")
            print(f"标签分布: {torch.bincount(labels)}")
            break
