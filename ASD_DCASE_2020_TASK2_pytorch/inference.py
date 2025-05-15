import torch
import numpy as np
from pathlib import Path
from parameters import params
from data_generator import get_h5_dataloaders
from models import ASDmodel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score


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
    """
    加载模型权重，对test集推理并计算AUC和pAUC。
    """
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
    print(f"Test AUC: {auc_score:.4f}, pAUC: {pauc_score:.4f}")
    # 自动寻找最佳阈值
    best_th, best_f1 = find_best_threshold(all_errors, all_labels)
    print(f"最佳F1阈值: {best_th:.6f}, 最佳F1: {best_f1:.4f}")
    return auc_score, pauc_score, best_th, best_f1


if __name__ == "__main__":
    # 1. 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 加载数据
    dataloaders = get_h5_dataloaders(params)
    test_loader = dataloaders['test']

    # 3. 获取特征维度
    sample_data, _ = next(iter(test_loader))
    feature_dim = sample_data.shape[1]

    # 4. 加载模型
    model = ASDmodel(input_dim=feature_dim).to(device)

    # 5. 指定模型权重路径
    # 请修改为你实际的模型权重路径
    model_path = '/root/CODE/ASD_DCASE_2020_TASK2_pytorch/checkpoints/Autoencoder__20250516_001629/model_best.pth'  # 修改为实际路径

    # 6. 推理与评估
    inference(model, test_loader, device, model_path=model_path) 