import torch
import utils
import pickle
import numpy as np
import os
from parameters import params
from pathlib import Path
from extract_features import FeatureExtractor
from data_generator import get_h5_dataloaders
from models import ASDmodel
from loss import AutoEncoderLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict
import csv
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score


def train_epoch(model: torch.nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                device: torch.device) -> float:
    """训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        device: 设备
        
    Returns:
        float: 平均训练损失
    """
    model.train()
    total_loss = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)  # 数据已经在正确的形状 [batch_size, feature_dim]
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def eval_epoch(model: torch.nn.Module,
               val_loader: DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device) -> Dict[str, float]:
    """评估一个epoch
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        loss_fn: 损失函数
        device: 设备
        
    Returns:
        Dict[str, float]: 包含验证指标的字典
    """
    model.eval()
    total_loss = 0
    normal_errors = []
    anomaly_errors = []
    
    # 使用tqdm显示进度条
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for data, labels in pbar:
            data = data.to(device)  # 数据已经在正确的形状 [batch_size, feature_dim]
            labels = labels.to(device)
            
            output = model(data)
            loss = loss_fn(output, data)
            total_loss += loss.item()
            
            # 计算重构误差
            errors = torch.mean((output - data) ** 2, dim=1)
            
            # 分别计算正常和异常样本的误差
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            
            if normal_mask.any():
                normal_errors.extend(errors[normal_mask].cpu().numpy())
            if anomaly_mask.any():
                anomaly_errors.extend(errors[anomaly_mask].cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算平均误差
    normal_error = np.mean(normal_errors) if normal_errors else 0
    anomaly_error = np.mean(anomaly_errors) if anomaly_errors else 0
    error_gap = anomaly_error - normal_error
    
    return {
        'val_loss': total_loss / len(val_loader),
        'normal_error': normal_error,
        'anomaly_error': anomaly_error,
        'error_gap': error_gap
    }

def main():
    # Set up directories for storing model checkpoints, predictions(result dir), and create a summary writer
    checkpoints_folder, result_dir, summary_writer = utils.setup(params)

    # Feature extraction（只需执行一次，后续可注释掉）
    feature_extractor = FeatureExtractor(params)
    feature_extractor.extract_and_save_all_h5()  # 用h5py方案

    # Set up train and eval data iterator
    dataloaders = get_h5_dataloaders(params)  # 用h5py的dataloader
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # 获取特征维度
    sample_data, _ = next(iter(train_loader))
    feature_dim = sample_data.shape[1]  # [batch_size, feature_dim]
    print(f"特征维度: {feature_dim}")

    # Create model, optimizer, and loss function
    model = ASDmodel(input_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    loss_fn = AutoEncoderLoss()

    start_epoch = 0
    min_val_loss = float('inf')
    patience = params.get('early_stopping_patience', 10)  # 你可以在params里设置
    patience_counter = 0

    # loss历史文件
    loss_history_path = os.path.join(checkpoints_folder, 'loss_history.csv')
    # 如果文件不存在，写表头
    if not os.path.exists(loss_history_path):
        with open(loss_history_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])

    # 使用tqdm显示epoch进度
    epoch_pbar = tqdm(range(start_epoch, params['epochs']), desc='Epochs')
    for epoch in epoch_pbar:
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        # 只保存当前最优模型，覆盖旧的
        if val_loss['val_loss'] < min_val_loss:
            min_val_loss = val_loss['val_loss']
            torch.save(model.state_dict(), os.path.join(checkpoints_folder, 'model_best.pth'))
            patience_counter = 0  # 有提升，重置计数
        else:
            patience_counter += 1  # 没提升，计数+1

        # 追加保存loss到csv
        with open(loss_history_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss['val_loss']])

        # 更新epoch进度条
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss["val_loss"]:.4f}',
            'best_val_loss': f'{min_val_loss:.4f}',
            'patience': patience_counter
        })

        # 满足早停条件，提前终止
        if patience_counter >= patience:
            print(f"早停触发！验证集损失连续 {patience} 轮未提升，提前终止训练。")
            break

    # 训练循环结束后，自动推理评估
    def find_best_threshold(errors, labels):
        thresholds = np.linspace(errors.min(), errors.max(), 1000)
        best_score = -1
        best_threshold = None
        for th in thresholds:
            preds = (errors > th).astype(int)
            score = f1_score(labels, preds)
            if score > best_score:
                best_score = score
                best_threshold = th
        return best_threshold, best_score

    def inference(model, test_loader, device, model_path=None, pauc_fpr=0.1):
        if model_path is not None:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"已加载模型权重: {model_path}")
        model.eval()
        all_errors = []
        all_labels = []
        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc='Test Inference'):
                data = data.to(device)
                labels = labels.cpu().numpy()
                output = model(data)
                errors = torch.mean((output - data) ** 2, dim=1).cpu().numpy()
                all_errors.extend(errors)
                all_labels.extend(labels)
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels)
        auc_score = roc_auc_score(all_labels, all_errors)
        fpr, tpr, thresholds = roc_curve(all_labels, all_errors)
        idx = np.where(fpr <= pauc_fpr)[0]
        pauc_score = auc(fpr[idx], tpr[idx]) / pauc_fpr if len(idx) > 1 else 0.0
        best_th, best_f1 = find_best_threshold(all_errors, all_labels)
        print(f"Test AUC: {auc_score:.4f}, pAUC: {pauc_score:.4f}")
        print(f"最佳F1阈值: {best_th:.6f}, 最佳F1: {best_f1:.4f}")
        return auc_score, pauc_score, best_th, best_f1

    # 加载最优模型并推理
    best_model_path = os.path.join(checkpoints_folder, 'model_best.pth')
    inference(model, test_loader, device, model_path=best_model_path)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()